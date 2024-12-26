from yacs.config import CfgNode as CN


_C = CN()


_C.SYSTEM = CN() # system settings

_C.SYSTEM.NUM_GPUS = 3
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.RANDOM_SEED = 2204
_C.SYSTEM.CONFIG_PATH = '~/ImageCLEFmed_MEDVQA-GI-2024-MMCP-Team/src/config/experiments'  # path to your main folder with .yaml files


_C.ACCELERATOR = CN() # Accelerator settings

_C.ACCELERATOR.SPLIT_BATCHES = False
_C.ACCELERATOR.ACCUMULATION_STEPS = 1
_C.ACCELERATOR.MIXED_PRECISION = 'no'
_C.ACCELERATOR.LOG_WITH = 'wandb'


_C.WANDB = CN() # wandb project and run names

_C.WANDB.PROJECT_NAME = 'MEDVQA-GI-2024-MMCP-Team'
_C.WANDB.RUN_NAME = 'Test Run'


_C.OPTIMIZER = CN() # torch.optim.AdamW optimizer settings

_C.OPTIMIZER.LEARNING_RATE = 1e-4
_C.OPTIMIZER.ADAM_BETA1 = 0.9
_C.OPTIMIZER.ADAM_BETA2 = 0.999
_C.OPTIMIZER.ADAM_WEIGHT_DECAY = 0.0
_C.OPTIMIZER.ADAM_EPSILON = 1e-8
_C.OPTIMIZER.USE_8_BIT_ADAM = False


_C.SCHEDULER = CN() # scheduler settings

_C.SCHEDULER.NAME = 'constant'
_C.SCHEDULER.WARMUP_STEPS = 0
_C.SCHEDULER.PREDICTION_TYPE = 'epsilon'
_C.SCHEDULER.SNR_GAMMA = -1


_C.LORA = CN() # LoRa settings - parameters for peft.LoraConfig

_C.LORA.RANK = 128
_C.LORA.ALPHA = 128
_C.LORA.HIDDEN_SIZE = 2048
_C.LORA.MAX_GRADIENT_NORM = 1.0
_C.LORA.USE_DORA = False
_C.LORA.USE_RSLORA = False
_C.LORA.DROPOUT = 0.0


_C.MSDM_UNET = CN() # MSDM Unet parameters

_C.MSDM_UNET.CHANNELS = 8
_C.MSDM_UNET.DROPOUT = 0.1
_C.MSDM_UNET.TIME_EMBEDDER_DIM = 1024
_C.MSDM_UNET.SELF_CONDITIONING = False
_C.MSDM_UNET.COND_EMB_BIAS = False


_C.MSDM_VAE = CN() # MSDM VAE parameters

_C.MSDM_VAE.CHANNELS = 3
_C.MSDM_VAE.EMB_CHANNELS = 8


_C.PATHS = CN() # all paths to required files 

_C.PATHS.LOCAL_FILES_ONLY = False

# paths to clef dataset parts

_C.PATHS.CLEF_DATASET_IMAGES_PATH = '~/datasets/clef/train/images'
_C.PATHS.CLEF_DATASET_TEXTS_TRAIN_PATH = '~/datasets/clef/train/train_captions.csv'
_C.PATHS.CLEF_DATASET_TEXTS_VALID_PATH = '~/datasets/clef/train/valid_captions.csv'
_C.PATHS.CLEF_DATASET_ALL_TEXTS_PATH = '~/datasets/clef/train/prompt-gt.csv'
_C.PATHS.CLEF_DATASET_TEST_PATH = '~/datasets/clef/train/prompt-gt.csv'

# paths to huggingface models

_C.PATHS.KANDINSKY2_PRIOR_PATH = 'kandinsky-community/kandinsky-2-2-prior'
_C.PATHS.KANDINSKY2_DECODER_PATH = 'kandinsky-community/kandinsky-2-2-decoder'
_C.PATHS.KANDINSKY3_PATH = 'kandinsky-community/kandinsky-3'

# paths to save checkpoints

_C.PATHS.KANDINSKY2_PRIOR_LORA_WEIGHTS_DIR = '~/ImageCLEFmed_MEDVQA-GI-2024-MMCP-Team/checkpoints/kandinsky2_prior'
_C.PATHS.KANDINSKY2_PRIOR_LORA_WEIGHTS_SUBFOLDER = 'prior'
_C.PATHS.KANDINSKY2_DECODER_LORA_WEIGHTS_DIR = '~/ImageCLEFmed_MEDVQA-GI-2024-MMCP-Team/checkpoints/kandinsky2_decoder'
_C.PATHS.KANDINSKY2_DECODER_LORA_WEIGHTS_SUBFOLDER = 'decoder'
_C.PATHS.KANDINSKY3_LORA_WEIGHTS_DIR = '~/ImageCLEFmed_MEDVQA-GI-2024-MMCP-Team/checkpoints/kandinsky3'
_C.PATHS.KANDINSKY3_LORA_WEIGHTS_SUBFOLDER = 'unet'

# paths to save generated images

_C.PATHS.RESULTING_IMAGES_FOLDER = '~/ImageCLEFmed_MEDVQA-GI-2024-MMCP-Team/images'
_C.PATHS.RESULTING_IMAGES_SUBFOLDER = 'new_images'


_C.TRAIN = CN() # training and validation parameters

_C.TRAIN.NUM_EPOCHS = 10

_C.TRAIN.IMAGE_VALIDATION_EPOCHS = 5
_C.TRAIN.FID_VALIDATION_EPOCHS = 10

_C.TRAIN.TRAIN_BATCH_SIZE = 64
_C.TRAIN.FID_BATCH_SIZE = 100
_C.TRAIN.VAL_BATCH_SIZE = 64

_C.TRAIN.NUM_INFERENCE_STEPS = 150

_C.TRAIN.TRAIN_IMAGE_RESOLUTION = 256
_C.TRAIN.VAL_IMAGE_RESOLUTION = 256
_C.TRAIN.FID_IMAGE_RESOLUTION = 256
_C.TRAIN.IMAGE_PADDING = True
_C.TRAIN.DATALOADER_NUM_WORKERS = 2
_C.TRAIN.MAX_TRAIN_STEPS = -1

# additional settings

_C.TRAIN.LOAD_LORA_WEIGHTS = False
_C.TRAIN.LOAD_LORA_TO_PIPELINE = False
_C.TRAIN.SAVE_BEST_FID_CHECKPOINTS = False
_C.TRAIN.SAVE_ALL_CHECKPOINTS = False
_C.TRAIN.FID_VALIDATION = False
_C.TRAIN.GENERATE_TEST_IMAGES = False
_C.TRAIN.USE_PARAPHRASES = False


_C.TRAIN.DATASET_NAME = 'CLEF'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()