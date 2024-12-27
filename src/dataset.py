import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn

from torch.utils.data import Dataset


class Kandinsky2PriorDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 tokenizer,
                 image_processor,
                 prompt_column_name='Prompt',
                 image_column_name='Filename',
                 resolution=256):
        super().__init__()

        self.images_path = images_path
        self.texts_path = texts_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.resolution = resolution

        self.data = pd.read_csv(self.texts_path)

        self.texts = self.data[prompt_column_name].tolist()
        self.tokenized_texts = tokenizer(
            self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.image_files = self.data[image_column_name].tolist()

    def __len__(self):
        return len(self.tokenized_texts['input_ids'])

    def __getitem__(self, item):
        filename = self.image_files[item]
        if '.jpg' in filename:
            image = Image.open(f'{self.images_path}/{filename}').convert('RGB')
        else:
            image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')
        image = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        return {"clip_pixel_values": image,
                "text_input_ids": self.tokenized_texts['input_ids'][item],
                "text_mask": self.tokenized_texts['attention_mask'][item].bool()}

    @staticmethod
    def collate_fn(examples):
        clip_pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.contiguous_format).float()
        text_input_ids = torch.stack([example["text_input_ids"] for example in examples])
        text_mask = torch.stack([example["text_mask"] for example in examples])
        return {"clip_pixel_values": clip_pixel_values, "text_input_ids": text_input_ids, "text_mask": text_mask}


class Kandinsky2DecoderDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 image_processor,
                 resolution=256,
                 image_column_name='Filename',
                 padding=False,
                 random_seed=2204):
        super().__init__()

        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        self.image_processor = image_processor
        self.padding = padding

        self.data = pd.read_csv(self.texts_path)
        self.image_files = self.data[image_column_name].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        filename = self.image_files[item]
        if '.jpg' in filename:
            image = Image.open(f'{self.images_path}/{filename}').convert('RGB')
        else:
            image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')

        if self.padding:
            w, h = image.size
            pad_value = abs(w - h) // 2
            r = int(abs(w - h) % 2 != 0)
            if h < w:
                img_padding = (0, pad_value, 0, pad_value + r)
            else:
                img_padding = (pad_value, 0, pad_value + r, 0)
            transform = T.Compose([T.Pad(padding=img_padding),
                                   T.Resize(self.resolution),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform = T.Compose([T.Resize((self.resolution, self.resolution)),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        return {"pixel_values": transform(image).to(torch.float32),
                "clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)}

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        clip_pixel_values = torch.stack([example["clip_pixel_values"] for example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "clip_pixel_values": clip_pixel_values}


class Kandinsky3Dataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 tokenizer,
                 prompt_column_name='Prompt',
                 image_file_col='Filename',
                 resolution=256,
                 padding=False):
        super().__init__()

        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.padding = padding

        self.data = pd.read_csv(self.texts_path)

        self.texts = self.data[prompt_column_name].tolist()
        self.tokenized_texts = tokenizer(
            self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        self.image_files = self.data[image_file_col].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        filename = self.image_files[item]
        if '.jpg' in filename:
            image = Image.open(f'{self.images_path}/{filename}').convert('RGB')
        else:
            image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')
        if self.padding:
            w, h = image.size
            pad_value = abs(w - h) // 2
            r = int(abs(w - h) % 2 != 0)
            if h < w:
                img_padding = (0, pad_value, 0, pad_value + r)
            else:
                img_padding = (pad_value, 0, pad_value + r, 0)
            transform = T.Compose([T.Pad(padding=img_padding),
                                   T.Resize(self.resolution),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform = T.Compose([T.Resize((self.resolution, self.resolution)),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        image = transform(image)
        return {"pixel_values": image,
                "input_ids": self.tokenized_texts['input_ids'][item],
                "attention_mask": self.tokenized_texts['attention_mask'][item].bool()}

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}


class ClefFIDDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 prompt_column_name='Prompt',
                 image_file_col='Filename',
                 resolution=256,
                 padding=False,
                 random_seed=2204):
        super().__init__()

        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        self.padding = padding

        self.prompts_info = pd.read_csv(self.texts_path, header=0, delimiter=';')
        self.image_files = self.prompts_info[image_file_col].unique()
        self.texts = self.prompts_info[prompt_column_name].sample(2000).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        filename = self.image_files[item]
        if '.jpg' in filename:
            image = Image.open(f'{self.images_path}/{filename}').convert('RGB')
        else:
            image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')
        if self.padding:
            w, h = image.size
            pad_value = abs(w - h) // 2
            r = int(abs(w - h) % 2 != 0)
            if h < w:
                img_padding = (0, pad_value, 0, pad_value + r)
            else:
                img_padding = (pad_value, 0, pad_value + r, 0)
            transform = T.Compose([T.Pad(padding=img_padding),
                                   T.Resize((self.resolution, self.resolution)),
                                   T.ToTensor()])
        else:
            transform = T.Compose([T.Resize((self.resolution, self.resolution)),
                                   T.ToTensor()])
        image = transform(image)
        return {"images": image, "texts": self.texts[item]}


class ClefMsdmVAEDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 image_column_name='Filename',
                 resolution=256):
        super().__init__()

        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution

        self.data = pd.read_csv(self.texts_path)

        self.image_files = self.data[image_column_name].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        filename = self.image_files[item]
        image = Image.open(f'{self.images_path}/images/{filename}').convert('RGB')
        transform = T.Compose([T.CenterCrop(min(image.size)),
                               T.Resize(self.resolution),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        return {"pixel_values": transform(image).to(torch.float32)}

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}


class ClefMsdmUnetDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 tokenizer,
                 image_file_col='Filename',
                 prompt_column_name='Prompt',
                 paraphrases_col='Paraphrase',
                 resolution=64,
                 p_dropout=0,
                 add_paraphrase=False,
                 train=True,
                 p_paraphrase=0,
                 ):
        super().__init__()

        self.images_path = images_path
        self.texts_path = texts_path
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.p_dropout = p_dropout
        self.p_paraphrase = p_paraphrase
        self.add_paraphrase = add_paraphrase
        self.train = train

        empty_context = tokenizer(
            [''], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.empty_ids, self.empty_mask = empty_context['input_ids'][0], empty_context['attention_mask'][0].bool()

        self.data = pd.read_csv(self.texts_path)

        self.image_files = self.data[image_file_col].tolist()

        self.texts = self.data[prompt_column_name].tolist()
        self.paraphrases = self.data[paraphrases_col].tolist()

        if self.add_paraphrase:
            self.texts += self.paraphrases
            self.image_files *= 2

        self.tokenized_texts = tokenizer(
            self.texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        if not self.add_paraphrase:
            self.tokenized_paraphrases = tokenizer(
                    self.paraphrases, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )

    def __len__(self):
        return len(self.tokenized_texts['input_ids'])

    def __getitem__(self, item):
        filename = self.image_files[item]
        image = Image.open(f'{self.images_path}/images/{filename}').convert('RGB')
        drop_context = torch.rand(1) < self.p_dropout
        paraphrase = False
        if not drop_context and not self.add_paraphrase:
            if torch.rand(1) < self.p_paraphrase:
                paraphrase = True

        first_transform = T.RandomCrop if self.train else T.CenterCrop
        transform = T.Compose([first_transform(min(image.size)),
                               T.Resize(self.resolution),
                               T.RandomHorizontalFlip(p=0.5) if drop_context else nn.Identity(),
                               T.RandomVerticalFlip(p=0.5) if drop_context else nn.Identity(),
                               T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        image = transform(image)

        input_ids = self.tokenized_texts['input_ids'][item]
        attention_mask = self.tokenized_texts['attention_mask'][item].bool()
        if drop_context:
            input_ids = self.empty_ids
            attention_mask = self.empty_mask
        elif paraphrase:
            input_ids = self.tokenized_paraphrases['input_ids'][item]
            attention_mask = self.tokenized_paraphrases['attention_mask'][item].bool()
        return {"pixel_values": image,
                "input_ids": input_ids,
                "attention_mask": attention_mask}

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}
