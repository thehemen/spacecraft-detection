import os
import sys
import torch
import numpy as np
import ruamel.yaml
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, data_dir, extension='.jpg', transform=None):
        self.data_dir = data_dir
        self.image_files  = [f for f in os.listdir(data_dir) if f.endswith(extension)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files) 

    def __getitem__(self, index):
        image_name = f'{self.data_dir}/{self.image_files[index]}'
        image = read_image(image_name)

        if self.transform: 
            image = self.transform(image)

        return image

if __name__ == '__main__':
    image_path = 'images/'

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True)
    ])

    dataset = ImageDataset(image_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4)

    img_sum = 0.0
    sqr_sum = 0.0

    num_batches = 0

    for images in tqdm(dataloader):
        img_sum += torch.mean(images, dim=[0, 2, 3])
        sqr_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = img_sum / num_batches
    std = (sqr_sum / num_batches - mean ** 2) ** 0.5

    mean = mean.detach().numpy().tolist()
    std = std.detach().numpy().tolist()

    yaml = ruamel.yaml.YAML()

    with open('augment.yaml', 'r') as fp:
        data = yaml.load(fp)

    data['mean'] = mean
    data['std'] = std

    with open('augment.yaml', 'w') as f:
        yaml.dump(data, f)

    print(f'\nMean: {mean}.')
    print(f'Std: {std}.')
