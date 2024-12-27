import argparse
import math
import wandb
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import make_grid
from config.medfusion_config import get_cfg_defaults
from dataset import ClefMedfusionUnetDataset, ClefFIDDataset

from msdm.models.pipelines.diffusion_pipeline import DiffusionPipeline
from msdm.models.estimators.unet import UNet
from msdm.models.noise_schedulers.gaussian_scheduler import GaussianNoiseScheduler
from msdm.models.embedders.time_embedder import TimeEmbbeding
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


@torch.no_grad()
def count_image_metrics(cfg, accelerator, pipeline, generator, fid_dataloader):
    pr_metric = ImprovedPrecessionRecall(splits_real=1, splits_fake=1, normalize=True).to(accelerator.device)
    torchmetrics_fid = FrechetInceptionDistance(normalize=True).to(accelerator.device)
    for fid_batch in tqdm(fid_dataloader):
        torchmetrics_fid.update(fid_batch['images'], real=True)
        pr_metric.update(fid_batch['images'], real=True)

        fake_images = pipeline.sample(num_samples=fid_batch['images'].shape[0],
                                      height=cfg['TRAIN']['FID_IMAGE_RESOLUTION'],
                                      width=cfg['TRAIN']['FID_IMAGE_RESOLUTION'],
                                      prompts=fid_batch['texts'],
                                      generator=generator,
                                      steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                      guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE'])

        torchmetrics_fid.update(fake_images, real=False)
        pr_metric.update(fake_images, real=False)

    fid_value = float(torchmetrics_fid.compute())
    precision, recall = pr_metric.compute()

    del torchmetrics_fid
    del pr_metric

    return fid_value, precision, recall


def save_model_checkpoint(cfg, accelerator, pipeline):
    unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
    torch.save({'model_state_dict': unwrapped_unet.state_dict()},
               os.path.join(cfg['PATHS']['MSDM_UNET_CHECKPOINT_DIR'], cfg['PATHS']['MSDM_UNET_CHECKPOINT_FILE']))


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

    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True, model_max_length=64)

    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': cfg['MSDM_UNET']['TIME_EMBEDDER_DIM']  # stable diffusion uses 4*model_channels (model_channels is about 256)
    }

    unet = UNet(in_ch=cfg['MSDM_UNET']['CHANNELS'],
                out_ch=cfg['MSDM_UNET']['CHANNELS'],
                spatial_dims=2,
                hid_chs=[256, 256, 512, 1024],
                kernel_sizes=[3, 3, 3, 3],
                strides=[1, 2, 2, 2],
                time_embedder=time_embedder,
                time_embedder_kwargs=time_embedder_kwargs,
                deep_supervision=False,
                use_res_block=True,
                use_attention='diffusers',
                cond_emb_hidden_dim=text_encoder.projection_dim,
                dropout=cfg['MSDM_UNET']['DROPOUT'],
                cond_emb_bias=cfg['MSDM_UNET']['COND_EMB_BIAS'])

    if cfg['TRAIN']['LOAD_MSDM_UNET_FROM_CHECKPOINT']:
        unet_checkpoint = torch.load(os.path.join(cfg['PATHS']['MSDM_UNET_CHECKPOINT_DIR'], cfg['PATHS']['MSDM_UNET_CHECKPOINT_FILE']))
        unet.load_state_dict(unet_checkpoint['model_state_dict'])

    unet.to(accelerator.device, dtype=torch.float32)

    noise_scheduler = GaussianNoiseScheduler(timesteps=1000,
                                             beta_start=0.002,
                                             beta_end=0.02,
                                             schedule_strategy='scaled_linear')

    if cfg['TRAIN']['MSDM_VAE_MODEL_TYPE'] == 'VAE':
        vae = VAE(
            in_channels=cfg['MSDM_VAE']['CHANNELS'],
            out_channels=cfg['MSDM_VAE']['CHANNELS'],
            emb_channels=cfg['MSDM_VAE']['EMB_CHANNELS'],
            spatial_dims=2,
            hid_chs=[64, 128, 256, 512],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            deep_supervision=1,
            use_attention='none',
        )
    else:
        vae = VQVAE(
            in_channels=cfg['MSDM_VAE']['CHANNELS'],
            out_channels=cfg['MSDM_VAE']['CHANNELS'],
            emb_channels=cfg['MSDM_VAE']['EMB_CHANNELS'],
            num_embeddings=8192,
            spatial_dims=2,
            hid_chs=[64, 128, 256, 512],
            beta=1,
            deep_supervision=True,
            use_attention='none',
        )
    vae.eval()
    vae_checkpoint = torch.load(os.path.join(cfg['PATHS']['MSDM_VAE_CHECKPOINT_DIR'], cfg['PATHS']['MSDM_VAE_CHECKPOINT_FILE']))
    vae.load_state_dict(vae_checkpoint['state_dict'])
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

    train_dataset = ClefMedfusionUnetDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'],
                                             cfg['PATHS']['CLEF_DATASET_TEXTS_TRAIN_PATH'],
                                             tokenizer,
                                             train=True,
                                             resolution=cfg['TRAIN']['TRAIN_IMAGE_RESOLUTION'],
                                             p_dropout=cfg['TRAIN']['MSDM_CONTEXT_DROPOUT_P'],
                                             add_paraphrase=cfg['TRAIN']['MSDM_ADD_PARAPHRASES_TO_TEXTS'],
                                             p_paraphrase=cfg['TRAIN']['MSDM_CONTEXT_PARAPHRASE_P'])

    val_dataset = ClefMedfusionUnetDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'],
                                           cfg['PATHS']['CLEF_DATASET_TEXTS_VALID_PATH'],
                                           tokenizer,
                                           train=False,
                                           resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                           add_paraphrase=cfg['TRAIN']['MSDM_ADD_PARAPHRASES_TO_TEXTS'],
                                           p_paraphrase=cfg['TRAIN']['MSDM_CONTEXT_PARAPHRASE_P'])

    fid_dataset = ClefFIDDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'],
                                 cfg['PATHS']['CLEF_DATASET_ALL_TEXTS_PATH'],
                                 resolution=cfg['TRAIN']['FID_IMAGE_RESOLUTION'],
                                 padding=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=ClefMedfusionUnetDataset.collate_fn,
        batch_size=cfg['TRAIN']['TRAIN_BATCH_SIZE'],
        num_workers=cfg['TRAIN']['DATALOADER_NUM_WORKERS'],
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=ClefMedfusionUnetDataset.collate_fn,
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
    max_train_steps = cfg['TRAIN']['NUM_EPOCHS'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
            name=cfg['SCHEDULER']['NAME'],
            optimizer=optimizer,
            num_warmup_steps=cfg['SCHEDULER']['WARMUP_STEPS'] * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
            num_training_steps=max_train_steps * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg['WANDB']['PROJECT_NAME'], config=cfg, init_kwargs={"wandb": {"name": cfg['WANDB']['RUN_NAME']}})

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
                                 use_self_conditioning=cfg['MSDM_UNET']['SELF_CONDITIONING'],
                                 use_ema=False,
                                 classifier_free_guidance_dropout=0,  # Disable during training by setting to 0
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

    best_fid = 1e10
    for epoch in range(first_epoch, cfg['TRAIN']['NUM_EPOCHS']):
        pipeline.unet.train()

        train_epoch_loss = 0.0
        train_epoch_batch_sum = 0.0

        for train_batch in train_dataloader:

            train_step_loss = 0.0

            with accelerator.accumulate(pipeline.unet):
                loss = pipeline.step(train_batch['pixel_values'], train_batch['input_ids'], train_batch['attention_mask'])

                avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['TRAIN_BATCH_SIZE'])).mean()

                train_step_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

                train_epoch_loss += train_step_loss * train_batch['pixel_values'].shape[0]
                train_epoch_batch_sum += train_batch['pixel_values'].shape[0]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.unet.parameters(), cfg['LORA']['MAX_GRADIENT_NORM'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"step_loss": train_step_loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

        train_epoch_loss /= train_epoch_batch_sum
        accelerator.log({"Train Loss": train_epoch_loss}, step=epoch)

        if accelerator.is_main_process:
            pipeline.unet.eval()

            validation_loss = 0.0
            validation_batch_sum = 0.0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_loss = pipeline.step(val_batch['pixel_values'], val_batch['input_ids'], val_batch['attention_mask'])
                    avg_loss = accelerator.gather(val_loss.repeat(cfg['TRAIN']['VAL_BATCH_SIZE'])).mean()
                    validation_loss += avg_loss.item() * val_batch['pixel_values'].shape[0]
                    validation_batch_sum += val_batch['pixel_values'].shape[0]

            validation_loss /= validation_batch_sum
            log_dict = {"Val Loss": validation_loss}

            if epoch == 0 or (epoch + 1) % cfg['TRAIN']['IMAGE_VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['NUM_EPOCHS'] - 1:
                generator = torch.Generator(device=accelerator.device)
                generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])

                val_images = pipeline.sample(num_samples=9,
                                             height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                             width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                             prompts=val_texts,
                                             steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                             guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach()

                val_images_praphrased = pipeline.sample(num_samples=9,
                                                        height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                                        width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                                        prompts=val_text_paraphrased,
                                                        generator=generator,
                                                        steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                        guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach()

                polyp_images = pipeline.sample(num_samples=4,
                                               height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                               width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                               prompts=polyp_texts,
                                               steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                               guidance_scale=cfg['TRAIN']['GUIDANCE_SCALE']).detach()

                log_dict = {"Image validation": wandb.Image(make_grid(val_images, nrow=3), caption=';'.join(val_texts)),
                            "Image Paraphrased validation": wandb.Image(make_grid(val_images_praphrased, nrow=3), caption=';'.join(val_text_paraphrased)),
                            "Polyp Validation": wandb.Image(make_grid(polyp_images, nrow=2), caption=';'.join(polyp_texts))}

                if cfg['TRAIN']['FID_VALIDATION'] and (epoch == 0 or (epoch + 1) % cfg['TRAIN']['FID_VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['NUM_EPOCHS'] - 1):
                    fid_value, precision, recall = count_image_metrics(cfg, accelerator, pipeline, generator, fid_dataloader)

                    log_dict['FID'] = fid_value
                    log_dict['Precision'] = precision
                    log_dict['Recall'] = recall
                    log_dict['F1'] = 2 * (precision * recall) / (precision + recall)

                    if cfg['TRAIN']['SAVE_BEST_FID_CHECKPOINTS'] and fid_value < best_fid:
                        save_model_checkpoint(cfg, accelerator, pipeline)
                        best_fid = fid_value

                torch.cuda.empty_cache()

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(log_dict)

    accelerator.wait_for_everyone()
    # save_model_checkpoint(cfg, accelerator, pipeline)
    accelerator.end_training()


if __name__ == "__main__":
    main()
