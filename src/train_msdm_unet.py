import argparse
import math
import wandb
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed, gather, gather_object
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import VQModel
from tqdm import tqdm
from piq import FID, multi_scale_ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import T5EncoderModel, T5Tokenizer
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from requests.exceptions import ConnectionError
from torchvision.utils import make_grid
from config.medfusion_config import get_cfg_defaults
from dataset import ClefMedfusionUnetDataset, ClefMedfusionFIDDataset

from msdm.models.pipelines.diffusion_pipeline import DiffusionPipeline
from msdm.models.estimators.unet import UNet
from msdm.models.noise_schedulers.gaussian_scheduler import GaussianNoiseScheduler
from msdm.models.embedders.time_embedder import TimeEmbbeding
from msdm.models.embedders.cond_embedders import TextEmbedder, LabelEmbedder
from msdm.models.embedders.latent_embedders import VAE, VQVAE
from msdm.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        required=False,
        help="Wandb project name",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        required=False,
        help="Wandb run name",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=False,
        help=('Path to config file')
    )

    args = parser.parse_args()

    return args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def main():
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config_path:
        cfg.merge_from_file(cfg['SYSTEM']['CONFIG_PATH'] + args.config_path)
    cfg.freeze()

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
        mixed_precision=cfg['ACCELERATOR']['MIXED_PRECISION'],
        log_with=cfg['ACCELERATOR']['LOG_WITH'],
    )

    set_seed(cfg['SYSTEM']['RANDOM_SEED'])

    if cfg['EXPERIMENT']['TEXT_ENCODER_TYPE'] == 'CLIP':
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True, model_max_length=64)
    else:
        tokenizer = T5Tokenizer.from_pretrained('kandinsky-community/kandinsky-3', subfolder="tokenizer", local_files_only=True, model_max_length=64)
        text_encoder = T5EncoderModel.from_pretrained('kandinsky-community/kandinsky-3', subfolder="text_encoder", use_safetensors=True, variant="fp16", local_files_only=True)

    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': cfg['UNET']['TIME_EMBEDDER_DIM'] # stable diffusion uses 4*model_channels (model_channels is about 256)
    }

    unet = UNet(in_ch=cfg['UNET']['CHANNELS'],
                out_ch=cfg['UNET']['CHANNELS'],
                spatial_dims=2,
                hid_chs=[256, 256, 512, 1024],
                kernel_sizes=[3, 3, 3, 3],
                strides=[1, 2, 2, 2],
                time_embedder=time_embedder,
                time_embedder_kwargs = time_embedder_kwargs,
                deep_supervision=False,
                use_res_block=True,
                use_attention='diffusers',
                cond_emb_hidden_dim=cfg['EXPERIMENT']['TEXT_ENCODER_HIDDEN_DIM'],
                dropout=cfg['UNET']['DROPOUT'],
                cond_emb_bias=cfg['UNET']['COND_EMB_BIAS'])
    
    if cfg['EXPERIMENT']['LOAD_UNET_FROM_CHECKPOINT']:
        unet_checkpoint = torch.load(cfg['EXPERIMENT']['MEDFUSION_UNET_CHECKPOINT_PATH'])
        unet.load_state_dict(unet_checkpoint['model_state_dict'])
    
    unet.to(accelerator.device, dtype=torch.float32)

    noise_scheduler = GaussianNoiseScheduler(timesteps=1000,
                                             beta_start=0.002,
                                             beta_end=0.02,
                                             schedule_strategy='scaled_linear')

    if cfg['EXPERIMENT']['VAE_MODEL_TYPE'] == 'VAE':
        vae = VAE(
            in_channels=cfg['VAE']['CHANNELS'], 
            out_channels=cfg['VAE']['CHANNELS'], 
            emb_channels=cfg['VAE']['EMB_CHANNELS'],
            spatial_dims=2,
            hid_chs =[64, 128, 256, 512], 
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            deep_supervision=1,
            use_attention='none',
        )
    else:
        vae = VQVAE(
            in_channels=cfg['VAE']['CHANNELS'], 
            out_channels=cfg['VAE']['CHANNELS'], 
            emb_channels=cfg['VAE']['EMB_CHANNELS'],
            num_embeddings=8192,
            spatial_dims=2,
            hid_chs=[64, 128, 256, 512],
            beta=1,
            deep_supervision=True,
            use_attention = 'none',
        )
    vae.eval()
    vae_checkpoint = torch.load(os.path.join(cfg['EXPERIMENT']['MEDFUSION_VAE_OUTPUT_DIR'], cfg['EXPERIMENT']['MEDFUSION_VAE_SUBFOLDER'], cfg['EXPERIMENT']['VAE_CHECKPOINT_NAME']))
    vae.load_state_dict(vae_checkpoint['state_dict'])
    # vae = VQModel.from_pretrained('kandinsky-community/kandinsky-2-2-decoder', subfolder="movq", local_files_only=True)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_dataset = ClefMedfusionUnetDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'],
                                             tokenizer,
                                             csv_filename=cfg['EXPERIMENT']['PROMPTS_FILE_NAME'],
                                             resolution=cfg['TRAIN']['IMAGE_RESOLUTION'],
                                             p_dropout=cfg['TRAIN']['CONTEXT_DROPOUT_P'],
                                             add_paraphrase=cfg['TRAIN']['ADD_PARAPHRASES_TO_TEXTS'],
                                             p_paraphrase=cfg['TRAIN']['CONTEXT_PARAPHRASE_P'])
    val_dataset = ClefMedfusionUnetDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'],
                                           tokenizer,
                                           csv_filename=cfg['EXPERIMENT']['PROMPTS_FILE_NAME'],
                                           resolution=cfg['TRAIN']['IMAGE_RESOLUTION'],
                                           train=False,
                                           add_paraphrase=cfg['TRAIN']['ADD_PARAPHRASES_TO_TEXTS'],
                                           p_paraphrase=cfg['TRAIN']['CONTEXT_PARAPHRASE_P'])
    fid_dataset = ClefMedfusionFIDDataset(cfg['EXPERIMENT']['DATASET_TRAIN_PATH'], resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'])
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg['TRAIN']['BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS'],
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg['TRAIN']['VAL_BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS'],
    )

    fid_dataloader = torch.utils.data.DataLoader(
        fid_dataset,
        shuffle=False,
        batch_size=cfg['TRAIN']['FID_BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS'],
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg['ACCELERATOR']['ACCUMULATION_STEPS'])
    max_train_steps = cfg['TRAIN']['EPOCHS'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
            name=cfg['SCHEDULER']['NAME'],
            optimizer=optimizer,
            num_warmup_steps=cfg['SCHEDULER']['WARMUP_STEPS'] * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
            num_training_steps=max_train_steps * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg['EXPERIMENT']['PROJECT_NAME'], config=cfg, init_kwargs={"wandb": {"name": cfg['EXPERIMENT']['NAME']}})

    unet, optimizer, train_dataloader, val_dataloader, fid_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, fid_dataloader, lr_scheduler
    )

    pipeline = DiffusionPipeline(noise_scheduler=noise_scheduler,
                                 unet=unet,
                                 tokenizer=tokenizer,
                                 latent_embedder=vae,
                                 text_encoder=text_encoder,
                                 estimator_objective='x_T',
                                 estimate_variance=False, 
                                 use_self_conditioning=cfg['UNET']['SELF_CONDITIONING'], 
                                 use_ema=False,
                                 classifier_free_guidance_dropout=0, # Disable during training by setting to 0
                                 do_input_centering=False,
                                 clip_x0=False,
                                 sample_every_n_steps=1000,
                                 device=accelerator.device
                                 )

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    val_texts = ['Generate an image with 1 finding.',
                 'Generate an image containing text.',
                 'Generate an image with an abnormality with the colors pink, red and white.',
                 'Generate an image not containing a green/black box artefact.',
                 'Generate an image containing a polyp.',
                 'Generate an image containing a green/black box artefact.',
                 'Generate an image with no polyps.',
                 'Generate an image with an an instrument located in the lower-right and lower-center.',
                 'Generate an image containing a green/black box artefact.']
    val_text_paraphrased = ['Create an image displaying a single abnormality.',
                            'Generate a medical image showing text.',
                            'Create a medical image displaying an anomaly characterized by shades of red and pink.',
                            'Create an image without any presence of an artifact in the form of a green or black box.',
                            'Create an image displaying a single polyp.',
                            'Create an image featuring a green/black box artifact.',
                            'Create a medical image showing an absence of polyps.',
                            'Create a medical image showcasing the use of a single medical tool located in the lower-right and lower-center.',
                            'Create a visual representation without showing any medical tools or equipment.']
    polyp_texts = ['generate an image containing a polyp',
                   'Create an image displaying a single polyp.',
                   'Create an image containing a single polyp.',
                   'Generate a picture displaying a polyp']
    val_epoch_loss = 0.0
    val_epoch_batch_sum = 0.0
    best_fid = 1e10
    for epoch in range(first_epoch, cfg['TRAIN']['EPOCHS']):
        pipeline.unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipeline.unet):
                loss = pipeline.step(batch['pixel_values'], batch['input_ids'], batch['attention_mask'])

                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['BATCH_SIZE'])).mean()
                train_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']
                val_epoch_loss += train_loss * batch['pixel_values'].shape[0]
                val_epoch_batch_sum += batch['pixel_values'].shape[0]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.unet.parameters(), cfg['LORA']['MAX_GRADIENT_NORM'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"medfusion_unet_step_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    if (global_step) % cfg['TRAIN']['IMAGE_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps:
                        try:
                            pipeline.unet.eval()
                            validation_loss = 0.0
                            validation_batch_sum = 0.0
                            with torch.no_grad():
                                for val_batch in val_dataloader:
                                    val_loss = pipeline.step(val_batch['pixel_values'], val_batch['input_ids'], val_batch['attention_mask'])
                                    avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['VAL_BATCH_SIZE'])).mean()
                                    validation_loss += avg_loss.item() * val_batch['pixel_values'].shape[0]
                                    validation_batch_sum += val_batch['pixel_values'].shape[0]
                            validation_loss /= validation_batch_sum 
                            
                            generator = torch.Generator(device=accelerator.device)
                            generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])
                            
                            val_images = pipeline.sample(num_samples=9,
                                                             img_size=[256, 256],
                                                             prompts=val_texts,
                                                             steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                             guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach()
                            
                            val_images_praphrased = pipeline.sample(num_samples=9,
                                                                        img_size=[256, 256],
                                                                        prompts=val_text_paraphrased,
                                                                        generator=generator,
                                                                        steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                                        guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach()
                            
                            polyp_images = pipeline.sample(num_samples=4,
                                                               img_size=[256, 256],
                                                               prompts=polyp_texts,
                                                               steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                               guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach()
                            
                            log_dict = {"Medfusion Unet validation": wandb.Image(make_grid(val_images, nrow=3), caption=';'.join(val_texts)),
                                        "Medfusion Unet Paraphrased validation": wandb.Image(make_grid(val_images_praphrased, nrow=3), caption=';'.join(val_text_paraphrased)),
                                        "Polyp Validation": wandb.Image(make_grid(polyp_images, nrow=2), caption=';'.join(polyp_texts)),
                                        "Train Loss:": val_epoch_loss / val_epoch_batch_sum,
                                        "Val Loss": validation_loss}
                            
                            val_epoch_loss = 0.0
                            val_epoch_batch_sum = 0.0

                            if cfg['EXPERIMENT']['FID_VALIDATION'] and ((global_step) % cfg['TRAIN']['FID_VALIDATION_STEPS'] == 0 or global_step + 1 == max_train_steps):
                                pr_metric = ImprovedPrecessionRecall(splits_real=1, splits_fake=1, normalize=True).to(accelerator.device)
                                torchmetrics_fid = FrechetInceptionDistance(normalize=True).to(accelerator.device)
                                for fid_batch in tqdm(fid_dataloader):
                                    torchmetrics_fid.update(fid_batch['images'], real=True)
                                    pr_metric.update(fid_batch['images'], real=True)

                                    fake_images = pipeline.sample(num_samples=fid_batch['images'].shape[0],
                                                                  img_size=[256, 256],
                                                                  prompts=fid_batch['texts'],
                                                                  generator=generator,
                                                                  steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                                  guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE'])
                                    
                                    torchmetrics_fid.update(fake_images, real=False)
                                    pr_metric.update(fake_images, real=False)

                                fid_value = float(torchmetrics_fid.compute())
                                log_dict['FID'] = fid_value
                                precision, recall = pr_metric.compute()
                                log_dict['Precision'] = precision
                                log_dict['Recall'] = recall
                                del torchmetrics_fid
                                del pr_metric

                                if fid_value < best_fid:
                                    unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
                                    torch.save({'model_state_dict': unwrapped_unet.state_dict()},
                                                os.path.join(cfg['EXPERIMENT']['MEDFUSION_UNET_OUTPUT_DIR'], cfg['EXPERIMENT']['MEDFUSION_UNET_SUBFOLDER'], cfg['EXPERIMENT']['MEDFUSION_UNET_CKPT_NAME']))
                                    best_fid = fid_value
                                    print('Model saved')
                            
                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(log_dict)

                            pipeline.unet.train()

                        except ConnectionError as error:
                            print('Unable to Validate: Connection Error')
                        torch.cuda.empty_cache()

                global_step += 1
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break


    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unwrapped_unet = accelerator.unwrap_model(unet)
    #     torch.save({'state_dict': unwrapped_unet.state_dict()},
    #                 os.path.join(cfg['EXPERIMENT']['MEDFUSION_VAE_OUTPUT_DIR'], cfg['EXPERIMENT']['MEDFUSION_VAE_SUBFOLDER'], 'vae.pt'))
    accelerator.end_training()


if __name__ == "__main__":
    main()