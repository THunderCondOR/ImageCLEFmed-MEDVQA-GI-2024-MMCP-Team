import argparse
import math
import wandb
import numpy as np
import os

import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoPipelineForText2Image, DDPMScheduler, PriorTransformer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, PeftModel
from peft import get_peft_model
from transformers import (CLIPImageProcessor, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)

from config.config import get_cfg_defaults
from dataset import Kandinsky2PriorDataset, RocoFIDDataset, ClefFIDDataset
from torchmetrics_pr_recall import ImprovedPrecessionRecall


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
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_snr_loss(loss, snr_gamma, noise_scheduler, timesteps):
    if snr_gamma != -1:
        snr = compute_snr(noise_scheduler, timesteps)
        if noise_scheduler.config.prediction_type == "v_prediction":
            snr = snr + 1
        mse_loss_weights = (torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr)
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    return loss.mean()


@torch.no_grad()
def get_target(image_encoder, noise_scheduler, clip_pixel_values, clip_mean, clip_std):
    image_embeds = image_encoder(clip_pixel_values).image_embeds
    noise = torch.randn_like(image_embeds)
    bsz = image_embeds.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=image_embeds.device)
    timesteps = timesteps.long()
    image_embeds = (image_embeds - clip_mean) / clip_std
    noisy_latents = noise_scheduler.add_noise(image_embeds, noise, timesteps)
    target = image_embeds
    return noisy_latents, timesteps, target


@torch.no_grad()
def validation_epoch(cfg, accelerator, prior, val_dataloader, text_encoder, image_encoder, noise_scheduler, clip_mean, clip_std, weight_dtype):
    prior.eval()
    validation_epoch_loss = 0.0
    validation_batch_sum = 0.0
    for val_batch in val_dataloader:
        text_encoder_output = text_encoder(val_batch["text_input_ids"])
        prompt_embeds = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        noisy_latents, timesteps, target = get_target(image_encoder, noise_scheduler, val_batch["clip_pixel_values"].to(weight_dtype), clip_mean, clip_std)

        model_pred = prior(
            noisy_latents,
            timestep=timesteps,
            proj_embedding=prompt_embeds,
            encoder_hidden_states=text_encoder_hidden_states,
            attention_mask=val_batch["text_mask"],
        ).predicted_image_embedding

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = get_snr_loss(loss, cfg['SCHEDULER']['SNR_GAMMA'], noise_scheduler, timesteps)

        avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['VAL_BATCH_SIZE'])).mean()
        validation_epoch_loss += avg_loss.item() * val_batch['clip_pixel_values'].shape[0]
        validation_batch_sum += val_batch['clip_pixel_values'].shape[0]

    return validation_epoch_loss / validation_batch_sum
    

def training_epoch(cfg, accelerator, prior, lora_layers, train_dataloader, text_encoder, image_encoder, noise_scheduler, clip_mean, clip_std, weight_dtype, optimizer, lr_scheduler, progress_bar, global_step):
    prior.train()
    train_epoch_loss = 0.0
    train_epoch_batch_sum = 0.0
    for train_batch in train_dataloader:
        train_step_loss = 0.0
        with accelerator.accumulate(prior):
            with torch.no_grad():
                text_encoder_output = text_encoder(train_batch["text_input_ids"])
                prompt_embeds = text_encoder_output.text_embeds
                text_encoder_hidden_states = text_encoder_output.last_hidden_state
            
            noisy_latents, timesteps, target = get_target(image_encoder, noise_scheduler, train_batch["clip_pixel_values"].to(weight_dtype), clip_mean, clip_std)
            model_pred = prior(
                noisy_latents,
                timestep=timesteps,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=train_batch["text_mask"],
            ).predicted_image_embedding

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = get_snr_loss(loss, cfg['SCHEDULER']['SNR_GAMMA'], noise_scheduler, timesteps)

            avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['TRAIN_BATCH_SIZE'])).mean()
            train_step_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

            train_epoch_loss += train_step_loss * train_batch["text_input_ids"].shape[0]
            train_epoch_batch_sum += train_batch["text_input_ids"].shape[0]

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(lora_layers, cfg['LORA']['MAX_GRADIENT_NORM'])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            #accelerator.log({"prior_train_step_loss": train_step_loss}, step=global_step)
            global_step += 1

            logs = {"step_loss": train_step_loss, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    return train_epoch_loss / train_epoch_batch_sum, global_step


@torch.no_grad()
def count_image_metrics(cfg, accelerator, pipeline, generator, fid_dataloader):
    pr_metric = ImprovedPrecessionRecall(splits_real=1, splits_fake=1, normalize=True).to(accelerator.device)
    torchmetrics_fid = FrechetInceptionDistance(normalize=True).to(accelerator.device)
    for fid_batch in tqdm(fid_dataloader):
        torchmetrics_fid.update(fid_batch['images'], real=True)
        pr_metric.update(fid_batch['images'], real=True)

        fake_images = pipeline(fid_batch['texts'],
                               num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                               height=cfg['TRAIN']['FID_IMAGE_RESOLUTION'],
                               width=cfg['TRAIN']['FID_IMAGE_RESOLUTION'],
                               generator=generator,
                               output_type='pt').images
        fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)
        
        torchmetrics_fid.update(fake_images, real=False)
        pr_metric.update(fake_images, real=False)

    fid_value = float(torchmetrics_fid.compute())
    precision, recall = pr_metric.compute()

    del torchmetrics_fid
    del pr_metric

    return fid_value, precision, recall


def save_model_checkpoint(cfg, accelerator, prior):
    unwrapped_prior = accelerator.unwrap_model(prior)
    unwrapped_prior.save_pretrained(
            os.path.join(cfg['PATHS']['KANDINSKY2_PRIOR_LORA_WEIGHTS_DIR'], cfg['PATHS']['KANDINSKY2_PRIOR_LORA_WEIGHTS_SUBFOLDER']),
            safe_serialization=True
    )


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
    np.random.seed(cfg['SYSTEM']['RANDOM_SEED'])

    noise_scheduler = DDPMScheduler.from_pretrained(cfg['PATHS']['KANDINSKY2_DECODER_PATH'], subfolder="scheduler", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg['PATHS']['KANDINSKY2_PRIOR_PATH'], subfolder="image_encoder", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    image_processor = CLIPImageProcessor.from_pretrained(cfg['PATHS']['KANDINSKY2_PRIOR_PATH'],  subfolder="image_processor", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    text_encoder = CLIPTextModelWithProjection.from_pretrained(cfg['PATHS']['KANDINSKY2_PRIOR_PATH'], subfolder="text_encoder", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    tokenizer = CLIPTokenizer.from_pretrained(cfg['PATHS']['KANDINSKY2_PRIOR_PATH'], subfolder="tokenizer", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    prior = PriorTransformer.from_pretrained(cfg['PATHS']['KANDINSKY2_PRIOR_PATH'], subfolder="prior", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])

    image_encoder.requires_grad_(False)
    prior.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    prior.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    prior_lora_config = LoraConfig(
        r=cfg['LORA']['RANK'],
        lora_alpha=cfg['LORA']['ALPHA'],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        use_dora=cfg['LORA']['USE_DORA'],
        use_rslora=cfg['LORA']['USE_RSLORA']
    )

    prior = get_peft_model(prior, prior_lora_config)
    prior.print_trainable_parameters()

    for p in prior.parameters():
        if p.requires_grad:
            p.data = p.to(torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, prior.parameters())

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    if cfg['EXPERIMENT']['DATASET_NAME'] == 'ROCO':
        train_dataset = Kandinsky2PriorDataset(cfg['PATHS']['ROCO_DATASET_IMAGES_TRAIN_PATH'], cfg['PATHS']['ROCO_DATASET_TEXTS_TRAIN_PATH'], tokenizer, image_processor, resolution=cfg['TRAIN']['TRAIN_IMAGE_RESOLUTION'])
        val_dataset = Kandinsky2PriorDataset(cfg['PATHS']['ROCO_DATASET_IMAGES_VALID_PATH'], cfg['PATHS']['ROCO_DATASET_TEXTS_VALID_PATH'], tokenizer, image_processor, resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'])
        fid_dataset = RocoFIDDataset(cfg['PATHS']['ROCO_DATASET_IMAGES_VALID_PATH'], cfg['PATHS']['ROCO_DATASET_TEXTS_VALID_PATH'], resolution=cfg['TRAIN']['FID_IMAGE_RESOLUTION'])
    else:
        train_dataset = Kandinsky2PriorDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'], cfg['PATHS']['CLEF_DATASET_TEXTS_TRAIN_PATH'], tokenizer, image_processor, prompt_column_name='Prompt', image_column_name='Filename', resolution=cfg['TRAIN']['TRAIN_IMAGE_RESOLUTION'])
        val_dataset = Kandinsky2PriorDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'], cfg['PATHS']['CLEF_DATASET_TEXTS_VALID_PATH'], tokenizer, image_processor, prompt_column_name='Prompt', image_column_name='Filename', resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'])
        fid_dataset = ClefFIDDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'], cfg['PATHS']['CLEF_DATASET_ALL_TEXTS_PATH'], image_file_col='Filename', captions_col='Prompt', resolution=cfg['TRAIN']['FID_IMAGE_RESOLUTION'])
    
    def collate_fn(examples):
        clip_pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.contiguous_format).float()
        text_input_ids = torch.stack([example["text_input_ids"] for example in examples])
        text_mask = torch.stack([example["text_mask"] for example in examples])
        return {"clip_pixel_values": clip_pixel_values, "text_input_ids": text_input_ids, "text_mask": text_mask}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg['TRAIN']['TRAIN_BATCH_SIZE'],
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
    if cfg['TRAIN']['MAX_TRAIN_STEPS'] == -1:
        max_train_steps = cfg['TRAIN']['NUM_EPOCHS'] * num_update_steps_per_epoch
    else:
        max_train_steps = cfg['TRAIN']['MAX_TRAIN_STEPS']
    lr_scheduler = get_scheduler(
            name=cfg['SCHEDULER']['NAME'],
            optimizer=optimizer,
            num_warmup_steps=cfg['SCHEDULER']['WARMUP_STEPS'] * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
            num_training_steps=max_train_steps * cfg['ACCELERATOR']['ACCUMULATION_STEPS'],
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg['WANDB']['PROJECT_NAME'], config=cfg, init_kwargs={"wandb": {"name": cfg['WANDB']['RUN_NAME']}})

    clip_mean = prior.clip_mean.clone()
    clip_std = prior.clip_std.clone()
    prior, optimizer, train_dataloader, val_dataloader, fid_dataloader, lr_scheduler = accelerator.prepare(
        prior, optimizer, train_dataloader, val_dataloader, fid_dataloader, lr_scheduler
    )

    clip_mean = clip_mean.to(weight_dtype).to(accelerator.device)
    clip_std = clip_std.to(weight_dtype).to(accelerator.device)

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    val_texts_roco = ["KUB x-ray demonstrating the stone KUB: kidney, ureter, and bladder",
                 "CT chest axial view showing a huge ascending aortic aneurysm (*)",
                 "Early axial T2-weighted MRI.",
                 "DSA showing tight left internal carotid artery stenosis.Foot note: DSA, Digital Subtraction Angiography.",
                 "Neck and head computed tomography image showing left odontogenic infection.",
                 "Plain film showing metallic object in appendix",
                 "CT-scan abdomen showing the bezoar at the pylorus.",
                 "Immediate postoperative plain radiograph showing femoral head fixation",
                 "Barcode sign” seen in M-mode."]
    
    val_caption_roco = " ;\n".join([f'{number}) {caption}' for number, caption in enumerate(val_texts_roco)])

    val_texts_clef = ['Generate an image with 1 finding.',
                'Generate an image containing text.',
                'Generate an image with an abnormality with the colors pink, red and white.',
                'Generate an image not containing a green/black box artefact.',
                'Generate an image containing a polyp.',
                'Generate an image containing a green/black box artefact.',
                'Generate an image with no polyps.',
                'Generate an image with an an instrument located in the lower-right and lower-center.',
                'Generate an image containing a green/black box artefact.']
    
    val_texts_clef_paraphrased = ['Create an image displaying a single abnormality.',
                            'Generate a medical image showing text.',
                            'Create a medical image displaying an anomaly characterized by shades of red and pink.',
                            'Create an image without any presence of an artifact in the form of a green or black box.',
                            'Create an image displaying a single polyp.',
                            'Create an image featuring a green/black box artifact.',
                            'Create a medical image showing an absence of polyps.',
                            'Create a medical image showcasing the use of a single medical tool located in the lower-right and lower-center.',
                            'Create a visual representation without showing any medical tools or equipment.']
    
    polyp_text_clef = ['generate an image containing a polyp']

    val_caption_clef = " ;\n".join([f'{number}) {caption}' for number, caption in enumerate(val_texts_clef)])
    val_clef_paraphrased_caption = " ;\n".join([f'{number}) {caption}' for number, caption in enumerate(val_texts_clef_paraphrased)])

    best_fid = 1e10

    for epoch in range(first_epoch, cfg['TRAIN']['NUM_EPOCHS']):
        train_loss, global_step = training_epoch(cfg, accelerator, prior, lora_layers, train_dataloader, text_encoder, image_encoder, noise_scheduler, clip_mean, clip_std, weight_dtype, optimizer, lr_scheduler, progress_bar, global_step)
        accelerator.log({"Train Loss": train_loss}, step=epoch)
        
        if accelerator.is_main_process:
            val_loss = validation_epoch(cfg, accelerator, prior, val_dataloader, text_encoder, image_encoder, noise_scheduler, clip_mean, clip_std, weight_dtype)

            log_dict = {"Val Loss": val_loss}
            
            if epoch == 0 or (epoch + 1) % cfg['TRAIN']['IMAGE_VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['NUM_EPOCHS'] - 1:
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    cfg['PATHS']['KANDINSKY2_DECODER_PATH'],
                    prior_prior=accelerator.unwrap_model(prior),
                    prior_text_encoder=text_encoder,
                    prior_image_encoder=image_encoder,
                    prior_image_processor=image_processor,
                    prior_tokenizer=tokenizer,
                    torch_dtype=weight_dtype,
                    local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY']
                )
                if cfg['EXPERIMENT']['LOAD_LORA_TO_PIPELINE']:
                    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, cfg['PATHS']['KANDINSKY2_DECODER_LORA_WEIGHTS_DIR'], subfolder=cfg['PATHS']['KANDINSKY2_DECODER_LORA_WEIGHTS_SUBFOLDER'])
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                generator = torch.Generator(device=accelerator.device)
                generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])

                if cfg['EXPERIMENT']['DATASET_NAME'] == 'ROCO':
                    val_images = pipeline(val_texts_roco, num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'], height=256, width=256, generator=generator).images
                    log_dict["Image validation"] = wandb.Image(image_grid(val_images, 3, 3), caption=val_caption_roco)
                else:
                    val_images = pipeline(val_texts_clef, num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'], height=256, width=256, generator=generator).images
                    val_images_paraphrased = pipeline(val_texts_clef_paraphrased,
                                                    num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                    height=256,
                                                    width=256,
                                                    generator=generator).images
                    polyp_images = pipeline(polyp_text_clef,
                                            num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                            height=256,
                                            width=256,
                                            num_images_per_prompt=9,
                                            generator=generator).images

                    log_dict["Image validation"] = wandb.Image(image_grid(val_images, 3, 3), caption=val_caption_clef)
                    log_dict["Image Paraphrased validation"] = wandb.Image(image_grid(val_images_paraphrased, 3, 3), caption=val_clef_paraphrased_caption)
                    log_dict["Polyp validation"] = wandb.Image(image_grid(polyp_images, 3, 3), caption=polyp_text_clef[0])

                if cfg['EXPERIMENT']['FID_VALIDATION'] and (epoch == 0 or (epoch + 1) % cfg['TRAIN']['FID_VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['NUM_EPOCHS'] - 1):
                    fid_value, precision, recall = count_image_metrics(cfg, accelerator, pipeline, generator, fid_dataloader)

                    log_dict['FID'] = fid_value
                    log_dict['Precision'] = precision
                    log_dict['Recall'] = recall
                    log_dict['F1'] = 2 * (precision * recall) / (precision + recall)
                    
                    if cfg['EXPERIMENT']['SAVE_BEST_FID_CHECKPOINTS'] and fid_value < best_fid:
                        save_model_checkpoint(cfg, accelerator, prior)
                        best_fid = fid_value

                del pipeline
                torch.cuda.empty_cache()
                            
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(log_dict)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_model_checkpoint(cfg, accelerator, prior)
    accelerator.end_training()


if __name__ == "__main__":
    main()
