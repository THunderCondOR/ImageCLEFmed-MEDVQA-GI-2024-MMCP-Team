import torch
import torch.nn as nn
import torch.nn.functional as F

from msdm.models.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, UnetResBlock, UnetBasicBlock
from msdm.loss.perceivers import LPIPS
from msdm.models.model_base import BasicModel

from pytorch_msssim import ssim


class DiagonalGaussianDistribution(nn.Module):
    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar) / batch_size

        return z, kl


class VAELoss:
    def __init__(self,
                 loss=torch.nn.L1Loss,
                 loss_kwargs={'reduction': 'none'},
                 perceiver=LPIPS,
                 perceiver_kwargs={},
                 embedding_loss_weight=1e-6,
                 perceptual_loss_weight=1.0,
                 device='cpu'
                 ):
        self.perceiver = perceiver(**perceiver_kwargs).to(device).eval()
        self.loss_fct = loss(**loss_kwargs)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight

    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth < 2):
            self.perceiver.eval()
            return self.perceiver(pred, target) * self.perceptual_loss_weight
        else:
            return 0

    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False, 
                      nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))

    def __call__(self, pred, pred_vertical, target, emb_loss):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0
        rec_loss = self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)
        # rec_loss = rec_loss/ torch.exp(self.logvar) + self.logvar # Note this is include in Stable-Diffusion but logvar is not used in optimizer
        loss += torch.sum(rec_loss)/pred.shape[0]

        for i, pred_i in enumerate(pred_vertical):
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)
            rec_loss_i = self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i]
            loss += torch.sum(rec_loss_i)/pred.shape[0]

        return loss + emb_loss * self.embedding_loss_weight


class VAE(BasicModel):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        spatial_dims=2,
        emb_channels=4,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        norm_name=("GROUP", {'num_groups': 8, "affine": True}),
        act_name=("Swish", {}),
        dropout=None,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        sample_every_n_steps=1000
    ):
        super().__init__()
        self.sample_every_n_steps = sample_every_n_steps
        # self.ssim_fct = SSIM(data_range=1, size_average=False, channel=out_channels, spatial_dims=spatial_dims, nonnegative_ssim=True)
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention] * len(strides)
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides

        # -------- Loss-Reg---------
        # self.logvar = nn.Parameter(torch.zeros(size=()) )

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims,
            in_channels,
            hid_chs[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            act_name=act_name,
            norm_name=norm_name,
            emb_channels=None
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[i-1],
                out_channels=hid_chs[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                downsample_kernel_size=downsample_kernel_sizes[i],
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2 * emb_channels, 3),
            BasicBlock(spatial_dims, 2 * emb_channels, 2 * emb_channels, 1)
        )


        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()  


        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name)

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[i+1],
                out_channels=hid_chs[i],
                kernel_size=kernel_sizes[i+1],
                stride=strides[i+1],
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])
        # self.logvar_ver = nn.ParameterList([
        #     nn.Parameter(torch.zeros(size=()) )
        #     for _ in range(1, deep_supervision+1)
        # ])

    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z

    def decode(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)

        return out, out_hor[::-1], emb_loss


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, emb_channels, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_channels = emb_channels
        self.beta = beta

        self.embedder = nn.Embedding(num_embeddings, emb_channels)
        self.embedder.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        assert z.shape[1] == self.emb_channels, "Channels of z and codebook don't match"
        z_ch = torch.moveaxis(z, 1, -1) # [B, C, *] -> [B, *, C]
        z_flattened = z_ch.reshape(-1, self.emb_channels) # [B, *, C] -> [Bx*, C], Note: or use contiguous() and view()

        # distances from z to embeddings e: (z - e)^2 = z^2 + e^2 - 2 e * z
        dist = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(self.embedder.weight**2, dim=1)
                -2 * torch.einsum("bd,dn->bn", z_flattened, self.embedder.weight.t()))  # [Bx*, num_embeddings]

        min_encoding_indices = torch.argmin(dist, dim=1)  # [Bx*]
        z_q = self.embedder(min_encoding_indices)  # [Bx*, C]
        z_q = z_q.view(z_ch.shape)  # [Bx*, C] -> [B, *, C]
        z_q = torch.moveaxis(z_q, -1, 1)  # [B, *, C] -> [B, C, *]

        # Compute Embedding Loss 
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss


class VQVAELoss:
    def __init__(self,
                 loss=torch.nn.L1Loss,
                 loss_kwargs={'reduction': 'none'},
                 perceiver=LPIPS,
                 perceiver_kwargs={},
                 embedding_loss_weight=1.0,
                 perceptual_loss_weight=1.0,
                 device='cpu'
                ):
        self.perceiver = perceiver(**perceiver_kwargs).to(device).eval()
        self.loss_fct = loss(**loss_kwargs)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight

    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth < 2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0

    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype) + 1) / 2, data_range=1, size_average=False,
                      nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))

    def __call__(self, pred, pred_vertical, target, emb_loss):
        interpolation_mode = 'nearest-exact'
        weights = [1/2**i for i in range(1+len(pred_vertical))]  # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w/tot_weight for w in weights]

        # Loss
        loss = 0
        loss += torch.mean(self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)) * weights[0]

        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            loss += torch.mean(self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)) * weights[i+1] 

        return loss + emb_loss * self.embedding_loss_weight


class VQVAE(BasicModel):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        spatial_dims=2,
        emb_channels=4,
        num_embeddings=8192,
        hid_chs=[32, 64, 128, 256],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        act_name=("Swish", {}),
        dropout=0.0,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        beta=0.25,
        sample_every_n_steps=1000

    ):
        super().__init__()
        self.sample_every_n_steps = sample_every_n_steps
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention] * len(strides)
        self.depth = len(strides)
        self.deep_supervision = deep_supervision

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(spatial_dims, in_channels, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0],
                             act_name=act_name, norm_name=norm_name)

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims,
                hid_chs[i-1],
                hid_chs[i],
                kernel_sizes[i],
                strides[i],
                kernel_sizes[i],
                norm_name,
                act_name,
                dropout,
                use_res_block,
                learnable_interpolation,
                use_attention[i])
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = BasicBlock(spatial_dims, hid_chs[-1], emb_channels, 1)


        # ----------- Quantizer --------------
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            emb_channels=emb_channels,
            beta=beta
        )    

        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name)

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims,
                hid_chs[i+1],
                hid_chs[i],
                kernel_size=kernel_sizes[i+1],
                stride=strides[i+1],
                upsample_kernel_size=strides[i+1],
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                skip_channels=0)
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True)
            for i in range(1, deep_supervision+1)
        ])

    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        return z

    def decode(self, z):
        z, _ = self.quantizer(z)
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None
            h = self.decoders[i](h)
        out = self.outc(h)

        return out, out_hor[::-1], emb_loss
