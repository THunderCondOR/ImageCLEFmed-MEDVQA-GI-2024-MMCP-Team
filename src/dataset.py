import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torch

from torch.utils.data import Dataset


class Kandinsky2PriorDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 tokenizer,
                 image_processor,
                 prompt_column_name='Caption',
                 image_column_name='ID',
                 resolution=256,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
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


class Kandinsky2DecoderDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 image_processor,
                 resolution=256,
                 image_column_name='ID',
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        self.image_processor = image_processor
        
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
        w, h = image.size
        pad_value = abs(w - h) // 2
        r = int(abs(w - h) % 2 != 0)
        if h < w:
            padding = (0, pad_value, 0, pad_value + r)
        else:
            padding = (pad_value, 0, pad_value + r, 0)
        transform = T.Compose([T.Pad(padding=padding),
                               T.Resize(self.resolution),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        return {"pixel_values": transform(image).to(torch.float32),
                "clip_pixel_values":  self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)}


class Kandinsky3Dataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 texts_file,
                 image_file_col='ID',
                 resolution=256,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        
        self.data = pd.read_csv(self.texts_path)

        self.hidden_satates = texts_file['hidden_states']
        self.attention_mask = texts_file['attention_mask']
        
        self.image_files = self.data[image_file_col].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        filename = self.image_files[item]
        if '.jpg' in filename:
            image = Image.open(f'{self.images_path}/{filename}').convert('RGB')
        else:
            image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')
        w, h = image.size
        pad_value = abs(w - h) // 2
        r = int(abs(w - h) % 2 != 0)
        if h < w:
            padding = (0, pad_value, 0, pad_value + r)
        else:
            padding = (pad_value, 0, pad_value + r, 0)
        transform = T.Compose([T.Pad(padding=padding),
                               T.Resize(self.resolution),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                               ])
        image = transform(image)
        return {"pixel_values": image,
                "attention_mask": torch.from_numpy(self.attention_mask[item]),
                "hidden_states": torch.from_numpy(self.hidden_satates[item])}


class RocoKandinsky3Dataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 tokenizer,
                 image_file_col='ID',
                 resolution=256,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        self.data = pd.read_csv(self.texts_path)
        
        self.texts = self.data['Caption'].tolist()
        self.tokenized_texts = tokenizer(
            self.texts, max_length=256, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.image_files = self.data[image_file_col].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        filename = self.image_files[item]
        image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')
        w, h = image.size
        pad_value = abs(w - h) // 2
        r = int(abs(w - h) % 2 != 0)
        if h < w:
            padding = (0, pad_value, 0, pad_value + r)
        else:
            padding = (pad_value, 0, pad_value + r, 0)
        transform = T.Compose([T.Pad(padding=padding),
                               T.Resize(self.resolution),
                               T.ToTensor(),
                               T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        image = transform(image)
        return {"pixel_values": image,
                "input_ids": self.tokenized_texts['input_ids'][item],
                "attention_mask": self.tokenized_texts['attention_mask'][item].bool()}
    

class FluxDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 texts_file,
                 image_file_col='ID',
                 resolution=256,
                 padding=True,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        self.padding = padding
        
        self.data = pd.read_csv(self.texts_path)

        self.clip_hidden_satates = texts_file['clip_hidden_states']
        self.t5_hidden_satates = texts_file['t5_hidden_states']
        
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
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
        else:
            transform = T.Compose([T.Resize((self.resolution, self.resolution)),
                                   T.ToTensor(),
                                   T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
        image = transform(image)
        return {"pixel_values": image,
                "clip_hidden_states": torch.from_numpy(self.clip_hidden_satates[item]),
                "t5_hidden_states": torch.from_numpy(self.t5_hidden_satates[item])}


class ClefFIDDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 image_file_col='ID',
                 captions_col='Caption',
                 resolution=256,
                 padding=True,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution
        self.padding = padding

        self.prompts_info = pd.read_csv(self.texts_path, header=0, delimiter=';')
        self.image_files = self.prompts_info[image_file_col].unique()
        self.texts = self.prompts_info[captions_col].sample(2000).tolist()

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
    

class RocoFIDDataset(Dataset):
    def __init__(self,
                 images_path,
                 texts_path,
                 image_file_col='ID',
                 captions_col='Caption',
                 resolution=256,
                 random_seed=2204):
        super().__init__()
        
        self.random_state = np.random.RandomState(random_seed)
        self.images_path = images_path
        self.texts_path = texts_path
        self.resolution = resolution

        self.prompts_info = pd.read_csv(self.texts_path)
        self.image_files = self.prompts_info[image_file_col].tolist()
        self.texts = self.prompts_info[captions_col].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        filename = self.image_files[item]
        if '.jpg' in filename:
            image = Image.open(f'{self.images_path}/{filename}').convert('RGB')
        else:
            image = Image.open(f'{self.images_path}/{filename}.jpg').convert('RGB')
        w, h = image.size
        pad_value = abs(w - h) // 2
        r = int(abs(w - h) % 2 != 0)
        if h < w:
            padding = (0, pad_value, 0, pad_value + r)
        else:
            padding = (pad_value, 0, pad_value + r, 0)
        transform = T.Compose([T.Pad(padding=padding),
                               T.Resize(self.resolution),
                               T.ToTensor()])
        image = transform(image)
        return {"images": image, "texts": self.texts[item]}