import argparse
import math
import wandb
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import Kandinsky3Pipeline, DDPMScheduler, VQModel, Kandinsky3UNet, VQModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from peft import LoraConfig
from peft import get_peft_model
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image

from config.config import get_cfg_defaults
from dataset import Kandinsky3Dataset, ClefFIDDataset
from torchmetrics_pr_recall import ImprovedPrecessionRecall
import h5py


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


def get_snr_loss(loss, snr_gamma, noise_scheduler, timesteps):
    if snr_gamma != -1:
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    loss = loss.mean()
    return loss


@torch.no_grad()
def get_noisy_latents(movq, noise_scheduler, pixel_values):
    latents = movq.encode(pixel_values.to(dtype=torch.float32)).latents #d * movq.config.scaling_factor

    noise = torch.randn_like(latents)

    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents, timesteps, noise, latents


@torch.no_grad()
def get_target(cfg, noise_scheduler, latents, noise, timesteps):
    if cfg['SCHEDULER']['PREDICTION_TYPE'] is not None:
        noise_scheduler.register_to_config(prediction_type=cfg['SCHEDULER']['PREDICTION_TYPE'])
    if noise_scheduler.config.prediction_type == "epsilon":
        return noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    

@torch.no_grad()
def validation_epoch(cfg, accelerator, unet, val_dataloader, movq, noise_scheduler):
    unet.eval()
    validation_loss = 0.0
    validation_batch_sum = 0.0
    for val_batch in val_dataloader:
        batch_size = val_batch["pixel_values"].shape[0]
        noisy_latents, timesteps, noise, latents = get_noisy_latents(movq, noise_scheduler, val_batch["pixel_values"])
        target = get_target(cfg, noise_scheduler, latents, noise, timesteps)

        encoded_text = val_batch["hidden_states"] * val_batch["attention_mask"].unsqueeze(2)

        model_pred = unet(noisy_latents, timesteps, encoded_text, val_batch["attention_mask"], return_dict=False)[0] #[:, :4]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = get_snr_loss(loss, cfg['SCHEDULER']['SNR_GAMMA'], noise_scheduler, timesteps)

        avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['VAL_BATCH_SIZE'])).mean()
        validation_loss += avg_loss.item() * batch_size
        validation_batch_sum += batch_size

    return validation_loss / validation_batch_sum


def training_epoch(cfg, accelerator, unet, lora_layers, train_dataloader, movq, noise_scheduler, optimizer, lr_scheduler, progress_bar, global_step):
    unet.train()
    train_epoch_loss = 0.0
    train_epoch_batch_sum = 0.0
    for train_batch in train_dataloader:
        train_step_loss = 0.0
        with accelerator.accumulate(unet):
            batch_size = train_batch["pixel_values"].shape[0]
            noisy_latents, timesteps, noise, latents = get_noisy_latents(movq, noise_scheduler, train_batch["pixel_values"])
            target = get_target(cfg, noise_scheduler, latents, noise, timesteps)

            encoded_text = train_batch["hidden_states"] * train_batch["attention_mask"].unsqueeze(2)

            model_pred = unet(noisy_latents, timesteps, encoded_text, train_batch["attention_mask"], return_dict=False)[0] #[:, :4]
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = get_snr_loss(loss, cfg['SCHEDULER']['SNR_GAMMA'], noise_scheduler, timesteps)

            avg_loss = accelerator.gather(loss.repeat(cfg['TRAIN']['TRAIN_BATCH_SIZE'])).mean()
            train_step_loss += avg_loss.item() / cfg['ACCELERATOR']['ACCUMULATION_STEPS']

            train_epoch_loss += train_step_loss * batch_size
            train_epoch_batch_sum += batch_size

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(lora_layers, cfg['LORA']['MAX_GRADIENT_NORM'])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            # accelerator.log({"kandinsky3_train_step_loss": train_step_loss}, step=global_step)
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
                               max_sequence_length=64,
                               output_type='pt').images
        fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)
        
        torchmetrics_fid.update(fake_images, real=False)
        pr_metric.update(fake_images, real=False)

    fid_value = float(torchmetrics_fid.compute())
    precision, recall = pr_metric.compute()

    del torchmetrics_fid
    del pr_metric

    return fid_value, precision, recall


def save_model_checkpoint(cfg, accelerator, unet):
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(
        os.path.join(cfg['PATHS']['KANDINSKY3_LORA_WEIGHTS_DIR'], cfg['PATHS']['KANDINSKY3_LORA_WEIGHTS_SUBFOLDER']),
        safe_serialization=False
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

    noise_scheduler = DDPMScheduler.from_pretrained(cfg['PATHS']['KANDINSKY3_PATH'], subfolder="scheduler", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    #tokenizer = T5Tokenizer.from_pretrained(cfg['PATHS']['KANDINSKY3_PATH'], subfolder="tokenizer", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    #text_encoder = T5EncoderModel.from_pretrained(cfg['PATHS']['KANDINSKY3_PATH'], subfolder="text_encoder", use_safetensors=True, variant="fp16", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    movq = VQModel.from_pretrained(cfg['PATHS']['KANDINSKY3_PATH'], subfolder="movq", use_safetensors=True, variant="fp16", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])
    unet = Kandinsky3UNet.from_pretrained(cfg['PATHS']['KANDINSKY3_PATH'], subfolder="unet", use_safetensors=True, variant="fp16", local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY'])

    unet.requires_grad_(False)
    movq.requires_grad_(False)
    #text_encoder.requires_grad_(False)

    for param in unet.parameters():
        param.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=cfg['LORA']['RANK'],
        lora_alpha=cfg['LORA']['ALPHA'],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    movq.to(accelerator.device, dtype=torch.float32)
    #text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.to(torch.float32)
    
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg['OPTIMIZER']['LEARNING_RATE'],
        betas=(cfg['OPTIMIZER']['ADAM_BETA1'], cfg['OPTIMIZER']['ADAM_BETA2']),
        weight_decay=cfg['OPTIMIZER']['ADAM_WEIGHT_DECAY'],
        eps=cfg['OPTIMIZER']['ADAM_EPSILON']
    )

    train_texts_file = h5py.File(cfg['PATHS']['KANDINSKY3_CLEF_PREENCODED_TRAIN'], 'r')
    valid_texts_file = h5py.File(cfg['PATHS']['KANDINSKY3_CLEF_PREENCODED_VALID'], 'r')
    
    train_dataset = Kandinsky3Dataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'], cfg['PATHS']['CLEF_DATASET_TEXTS_TRAIN_PATH'], texts_file=train_texts_file, image_file_col='Filename', resolution=cfg['TRAIN']['TRAIN_IMAGE_RESOLUTION'])
    val_dataset = Kandinsky3Dataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'], cfg['PATHS']['CLEF_DATASET_TEXTS_VALID_PATH'], texts_file=valid_texts_file, image_file_col='Filename', resolution=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'])
    fid_dataset = ClefFIDDataset(cfg['PATHS']['CLEF_DATASET_IMAGES_PATH'], cfg['PATHS']['CLEF_DATASET_ALL_TEXTS_PATH'], image_file_col='Filename', captions_col='Prompt', resolution=cfg['TRAIN']['FID_IMAGE_RESOLUTION']) 
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        hidden_states = torch.stack([example["hidden_states"] for example in examples])
        return {"pixel_values": pixel_values, "attention_mask": attention_mask, "hidden_states": hidden_states}

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

    val_texts = ['Generate an image with 1 finding.',
                'Generate an image containing text.',
                'Generate an image with an abnormality with the colors pink, red and white.',
                'Generate an image not containing a green/black box artefact.',
                'Generate an image containing a polyp.',
                'Generate an image containing a green/black box artefact.',
                'Generate an image with no polyps.',
                'Generate an image with an an instrument located in the lower-right and lower-center.',
                'Generate an image containing a green/black box artefact.']
    
    val_texts_paraphrased = ['Create an image displaying a single abnormality.',
                            'Generate a medical image showing text.',
                            'Create a medical image displaying an anomaly characterized by shades of red and pink.',
                            'Create an image without any presence of an artifact in the form of a green or black box.',
                            'Create an image displaying a single polyp.',
                            'Create an image featuring a green/black box artifact.',
                            'Create a medical image showing an absence of polyps.',
                            'Create a medical image showcasing the use of a single medical tool located in the lower-right and lower-center.',
                            'Create a visual representation without showing any medical tools or equipment.']
    
    polyp_text = ['generate an image containing a polyp']

    val_caption = " ;\n".join([f'{number}) {caption}' for number, caption in enumerate(val_texts)])
    val_paraphrased_caption = " ;\n".join([f'{number}) {caption}' for number, caption in enumerate(val_texts_paraphrased)])

    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(first_epoch, cfg['TRAIN']['NUM_EPOCHS']):
        train_loss, global_step = training_epoch(cfg, accelerator, unet, lora_layers, train_dataloader, movq, noise_scheduler, optimizer, lr_scheduler, progress_bar, global_step)
        accelerator.log({"Train Loss": train_loss}, step=epoch)

        if accelerator.is_main_process:
            val_loss = validation_epoch(cfg, accelerator, unet, val_dataloader, movq, noise_scheduler)
            log_dict = {"Val Loss": val_loss}
            
            if epoch == 0 or (epoch + 1) % cfg['TRAIN']['IMAGE_VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['NUM_EPOCHS'] - 1:
                pipeline = Kandinsky3Pipeline.from_pretrained(
                    cfg['PATHS']['KANDINSKY3_PATH'],
                    unet=accelerator.unwrap_model(unet),
                    scheduler=accelerator.unwrap_model(noise_scheduler),
                    movq=movq,
                    torch_dtype=weight_dtype,
                    variant="fp16",
                    use_safetensors=True,
                    local_files_only=cfg['PATHS']['LOCAL_FILES_ONLY']
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                
                generator = torch.Generator(device=accelerator.device)
                generator = generator.manual_seed(cfg['SYSTEM']['RANDOM_SEED'])

                val_images = pipeline(val_texts,
                                      num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                      height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                      width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                      max_sequence_length=64,
                                      generator=generator).images
                val_images_paraphrased = pipeline(val_texts_paraphrased,
                                                  num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                                  height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                                  width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                                  max_sequence_length=64,
                                                  generator=generator).images

                polyp_images = pipeline(polyp_text,
                                        num_inference_steps=cfg['TRAIN']['NUM_INFERENCE_STEPS'],
                                        height=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                        width=cfg['TRAIN']['VAL_IMAGE_RESOLUTION'],
                                        num_images_per_prompt=9,
                                        max_sequence_length=64,
                                        generator=generator).images

                log_dict["Image validation"] = wandb.Image(image_grid(val_images, 3, 3), caption=val_caption)
                log_dict["Image Paraphrased validation"] = wandb.Image(image_grid(val_images_paraphrased, 3, 3), caption=val_paraphrased_caption)
                log_dict["Polyp validation"] = wandb.Image(image_grid(polyp_images, 3, 3), caption=polyp_text[0])

                if cfg['EXPERIMENT']['FID_VALIDATION'] and (epoch == 0 or (epoch + 1) % cfg['TRAIN']['FID_VALIDATION_EPOCHS'] == 0 or epoch == cfg['TRAIN']['NUM_EPOCHS'] - 1):
                    fid_value, precision, recall = count_image_metrics(cfg, accelerator, pipeline, generator, fid_dataloader)

                    log_dict['FID'] = fid_value
                    log_dict['Precision'] = precision
                    log_dict['Recall'] = recall
                    log_dict['F1'] = 2 * (precision * recall) / (precision + recall)
                
                if cfg['EXPERIMENT']['SAVE_BEST_FID_CHECKPOINTS'] and fid_value < best_fid:
                    save_model_checkpoint(cfg, accelerator, unet)
                    best_fid = fid_value
                
                del pipeline
                torch.cuda.empty_cache()
                
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(log_dict)


    accelerator.wait_for_everyone()
    train_texts_file.close()
    valid_texts_file.close()
    if accelerator.is_main_process:
        save_model_checkpoint(cfg, accelerator, unet)
    accelerator.end_training()


if __name__ == "__main__":
    main()
