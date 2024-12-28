import torch
import torch.nn.functional as F

from msdm.models.model_base import BasicModel
from msdm.utils.train_utils import EMAModel


def kl_gaussians(mean1, logvar1, mean2, logvar2):
    """ Compute the KL divergence between two gaussians."""
    return 0.5 * (logvar2-logvar1 + torch.exp(logvar1 - logvar2) + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2) - 1.0)


def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])


class DiffusionPipeline(BasicModel):
    def __init__(self,
                 noise_scheduler,
                 unet,
                 tokenizer,
                 text_encoder,
                 latent_embedder=None,
                 estimator_objective='x_T',  # 'x_T' or 'x_0'
                 estimate_variance=False,
                 use_self_conditioning=False,
                 classifier_free_guidance_dropout=0.5,  # Probability to drop condition during training, has only an effect for label-conditioned training 
                 num_samples=4,
                 do_input_centering=True,  # Only for training
                 clip_x0=True,  # Has only an effect during traing if use_self_conditioning=True, import for inference/sampling  
                 use_ema=False,
                 ema_kwargs={},
                 loss=torch.nn.L1Loss,
                 loss_kwargs={},
                 sample_every_n_steps=1000,
                 device='cpu'
                 ):
        # self.save_hyperparameters(ignore=['noise_estimator', 'noise_scheduler'])
        super().__init__()
        self.loss_fct = loss(**loss_kwargs)
        self.sample_every_n_steps = sample_every_n_steps

        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.tokenizer = tokenizer

        self.latent_embedder = latent_embedder
        self.text_encoder = text_encoder

        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.num_samples = num_samples
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.do_input_centering = do_input_centering
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0
        self.device = device

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.unet, **ema_kwargs)

    def step(self, x_0, condition=None, condition_mask=None):
        results = {}
        # Embed into latent space or normalize
        if self.latent_embedder is not None:
            self.latent_embedder.eval()
            with torch.no_grad():
                x_0 = self.latent_embedder.encode(x_0)
        if self.do_input_centering:
            x_0 = 2*x_0-1  # [0, 1] -> [-1, 1]
        # if self.clip_x0:
        #     x_0 = torch.clamp(x_0, -1, 1)
        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0,T-1] and compute x_t (noisy version of x_0 at t)
            x_t, x_T, t = self.noise_scheduler.sample(x_0)
        # Use EMA Model
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.unet

        with torch.no_grad():
            condition_emb = self.text_encoder(condition, attention_mask=condition_mask, return_dict=False)[0]
        # Re-estimate x_T or x_0, self-conditioned on previous estimate
        self_cond = None
        if self.use_self_conditioning:
            with torch.no_grad():
                pred, pred_vertical = noise_estimator(x_t, t, condition_emb, condition_mask, None)
                if self.estimate_variance:
                    pred, _ = pred.chunk(2, dim=1)  # Seperate actual prediction and variance estimation
                if self.estimator_objective == "x_T":  # self condition on x_0
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0":  # self condition on x_T
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # Classifier free guidance
        if torch.rand(1) < self.classifier_free_guidance_dropout:
            condition = None

        # Run Denoise

        pred, pred_vertical = noise_estimator(x_t, t, condition_emb, condition_mask, self_cond)

        # Separate variance (scale) if it was learned
        if self.estimate_variance:
            pred, pred_var =  pred.chunk(2, dim=1)  # Separate actual prediction and variance estimation

        # Specify target
        if self.estimator_objective == "x_T":
            target = x_T
        elif self.estimator_objective == "x_0":
            target = x_0
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        loss = 0
        weights = [1/2**i for i in range(1+len(pred_vertical))] # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]
        # ----------------- MSE/L1, ... ----------------------
        loss += self.loss_fct(pred, target)*weights[0]

        # ----------------- Variance Loss --------------
        if self.estimate_variance:
            # var_scale = var_scale.clamp(-1, 1) # Should not be necessary
            var_scale = (pred_var+1)/2 # Assumed to be in [-1, 1] -> [0, 1]
            pred_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=var_scale)
            # pred_logvar = pred_var  # If variance is estimated directly

            if self.estimator_objective == 'x_T':
                pred_x_0 = self.noise_scheduler.estimate_x_0(x_t, x_T, t, clip_x0=self.clip_x0)
            elif self.estimator_objective == "x_0":
                pred_x_0 = pred
            else:
                raise NotImplementedError()

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, x_0, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=0)

            kl_loss = torch.mean(kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar), dim=list(range(1, x_0.ndim)))
            nnl_loss = torch.mean(F.gaussian_nll_loss(pred_x_0, x_0, torch.exp(pred_logvar), reduction='none'), dim=list(range(1, x_0.ndim)))
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss

            results['variance_scale'] = torch.mean(var_scale)
            results['variance_loss'] = var_loss

        # ----------------------------- Deep Supervision -------------------------
        for i, pred_i in enumerate(pred_vertical):
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)
            loss += self.loss_fct(pred_i, target_i)*weights[i+1]
        results['loss'] = loss

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            results['L2'] = F.mse_loss(pred, target)
            results['L1'] = F.l1_loss(pred, target)
            # results['SSIM'] = SSIMMetric(data_range=pred.max()-pred.min(), spatial_dims=source.ndim-2)(pred, target)

            # for i, pred_i in enumerate(pred_vertical):
            #     target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)
            #     results[f'L1_{i}'] = F.l1_loss(pred_i, target_i).detach()

        return loss

    def forward(self, x_t, t, condition_emb=None, condition_mask=None, self_cond=None, guidance_scale=1.0, cold_diffusion=False):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.unet
        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition_emb is not None) and (guidance_scale != 1.0):
            # Model prediction 
            pred, _ = noise_estimator(torch.cat([x_t] * 2), torch.cat([t] * 2), condition_embed=condition_emb, condition_mask=condition_mask, self_cond=self_cond)
            pred_uncond, pred_cond = pred.chunk(2)
            pred = (guidance_scale + 1.0) * pred_cond - guidance_scale * pred_uncond
            # pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond = pred_uncond.chunk(2, dim=1)
                pred_cond,   pred_var_cond = pred_cond.chunk(2, dim=1)
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            # condition_emb = self.text_encoder(condition, attention_mask=condition_mask, return_dict=False)[0]
            pred, _ =  noise_estimator(x_t, t, condition_emb=condition_emb, condition_mask=condition_mask, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var = pred.chunk(2, dim=1)

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5 # [-1, 1] -> [0, 1]
            pred_var_value = pred_var
        else:
            pred_var_scale = 0
            pred_var_value = None
        # pred_var_scale = pred_var_scale.clamp(0, 1)
        if self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T
        elif self.estimator_objective == 'x_T':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = pred
            self_cond = x_0
        else:
            raise ValueError("Unknown Objective")

        return x_t_prior, x_0, x_T, self_cond 

    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition_emb=None, condition_mask=None, guidance_scale=1.0, use_ddim=True, **kwargs):
        self_cond = None 
        # ---------- run denoise loop ---------------
        if use_ddim:
            steps = self.noise_scheduler.timesteps if steps is None else steps
            timesteps_array = torch.linspace(0, self.noise_scheduler.T-1, steps, dtype=torch.long, device=x_t.device)  # [0, 1, 2, ..., T-1] if steps = T 
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)]  # [0, ...,T-1] (target time not time of x_t)

        for i, t in enumerate(reversed(timesteps_array)):
            # UNet prediction 
            x_t, x_0, x_T, self_cond = self(x_t, t.expand(x_t.shape[0]), condition_emb, condition_mask, self_cond=self_cond, guidance_scale=guidance_scale, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None

            if use_ddim and (steps-i - 1 > 0):
                t_next = timesteps_array[steps-i-2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # ------ Eventually decode from latent space into image space--------
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)

        x_t = normalize(x_t).clamp(0, 1)
        return x_t # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, num_samples, height=256, width=256, prompts=None, generator=None, guidance_scale=1.0, **kwargs):
        condition, condition_mask = None, None
        uncond, uncond_mask = None, None
        if prompts is not None:
            self.text_encoder.eval()
            text_tokens = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            condition, condition_mask = text_tokens['input_ids'].to(self.device), text_tokens['attention_mask'].to(self.device)
            uncond_tokens = self.tokenizer([''] * num_samples, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            uncond, uncond_mask = uncond_tokens['input_ids'].to(self.device), uncond_tokens['attention_mask'].to(self.device)
            condition_emb = self.text_encoder(condition, attention_mask=condition_mask, return_dict=False)[0]
            uncond_emb = self.text_encoder(uncond, attention_mask=uncond_mask, return_dict=False)[0]
            condition_emb = torch.cat([uncond_emb, condition_emb])
            condition_mask = torch.cat([uncond_mask, condition_mask]).bool()
        x_T = self.noise_scheduler.x_final_deterministic((num_samples, self.unet.num_channels, height // 8, width // 8), generator=generator, device=self.device)
        x_0 = self.denoise(x_T, condition_emb=condition_emb, condition_mask=condition_mask, guidance_scale=guidance_scale, **kwargs)
        return x_0

    @torch.no_grad()
    def interpolate(self, img1, img2, i=None, condition=None, condition_mask=None, lam=0.5, **kwargs):
        assert img1.shape == img2.shape, "Image 1 and 2 must have equal shape"

        t = self.noise_scheduler.T-1 if i is None else i
        t = torch.full(img1.shape[:1], i, device=img1.device)

        img1_t = self.noise_scheduler.estimate_x_t(img1, t=t, clip_x0=self.clip_x0)
        img2_t = self.noise_scheduler.estimate_x_t(img2, t=t, clip_x0=self.clip_x0)

        img = (1 - lam) * img1_t + lam * img2_t
        img = self.denoise(img, i, condition, condition_mask, **kwargs)
        return img

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.unet)
