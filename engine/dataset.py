import cv2
import glob
import torch
from torch.utils.data import Dataset
from engine.misc import get_image_bbox

def read_label(filename):
    with open(filename, 'r') as f:
        content = f.read()
        vals = content.split(' ')
        class_id = int(vals[0])
        label = [float(val) for val in vals[1:]]

    return label

class SpaceshipDataset(Dataset):
    def __init__(self, dataset, indexes, transform=None, bboxes=None):
        img_dir = '/'.join(dataset['path'].split('/')[2:])
        label_dir = img_dir.replace('images', 'labels')

        self.img_names = sorted(glob.glob(f'{img_dir}/*.jpg'))
        self.label_names = sorted(glob.glob(f'{label_dir}/*.txt'))

        self.class_name = dataset['names'][0]
        self.indexes = indexes

        self.transform = transform
        self.bboxes = bboxes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_name = self.img_names[self.indexes[idx]]
        label_name = self.label_names[self.indexes[idx]]

        init_image = cv2.imread(img_name)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)

        if self.bboxes is not None:
            bbox = self.bboxes[idx]
            width, height = init_image.shape[:2][::-1]

            bbox, coeffs = get_image_bbox(bbox, width, height)
            x1, y1, x2, y2 = bbox

            init_image = init_image[y1: y2, x1: x2]

        init_label = read_label(label_name)
        init_label.append(self.class_name)

        if self.transform:
            try:
                transformed = self.transform(image=init_image, bboxes=[init_label])
            except Exception as exc:
                transformed = self.transform(image=init_image, bboxes=[init_label])

            image, label = transformed['image'], transformed['bboxes']

        if len(label) > 0:
            label = torch.tensor(label[0][:-1])
        else:
            label = torch.tensor(init_label[:-1])

        if self.bboxes is None:
            return image, label
        else:
            return image, label, coeffs
