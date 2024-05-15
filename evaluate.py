import cv2
import glob
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from engine.model import SpaceshipDetector
from engine.dataset import read_label
from engine.metric import get_iou_score
from engine.augmentation import get_augmentation_transform
from engine.cross_validation import get_cv_split
from engine.misc import get_predicted_bbox
from engine.misc import get_image_bbox
from engine.misc import get_aligned_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect a spaceship on the image.')
    parser.add_argument('--augment_cfg', default='augment.yaml')
    parser.add_argument('--model_name', default='model.pt')
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--bbox_coeff', type=float, default=5.0)
    parser.add_argument('--share', type=float, default=0.8)
    parser.add_argument('--image_path', default='images/')
    parser.add_argument('--class_name', default='Spaceship')
    args = parser.parse_args()

    with open(args.augment_cfg, 'r') as f:
        augment_cfg = yaml.load(f, Loader=yaml.CLoader)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}.')

    transform = get_augmentation_transform(augment_cfg, args.image_size, is_train=False)
    idx = get_cv_split(args.share, is_train=False)

    label_path = args.image_path.replace('images', 'labels')

    image_names = sorted(glob.glob(f'{args.image_path}/*.jpg'))
    label_names = sorted(glob.glob(f'{label_path}/*.txt'))

    image_names = [image_names[i] for i in idx]
    label_names = [label_names[i] for i in idx]

    model = SpaceshipDetector()
    model.load_state_dict(torch.load(args.model_name))

    model = model.to(device)
    model.eval()

    scores = []

    with tqdm(total=len(image_names), position=0, leave=True) as tqdm_bar:
        for image_name, label_name in zip(image_names, label_names):
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            width, height = image.shape[:2][::-1]

            label = read_label(label_name)
            label.append(args.class_name)

            transformed = transform(image=image, bboxes=[label])
            image_transformed = transformed['image']

            label_transformed = torch.tensor(transformed['bboxes'][0][:-1])[np.newaxis, ...]
            image_transformed = image_transformed[np.newaxis, ...]

            label_transformed = label_transformed.to(device)
            image_transformed = image_transformed.to(device)

            with torch.no_grad():
                pred = model(image_transformed)
                pred = get_predicted_bbox(pred, args.bbox_coeff)[0]

                pred, coeffs = get_image_bbox(pred, width, height)

                x1, y1, x2, y2 = pred
                coeffs = coeffs.to(device)

                image_cropped = image[y1: y2, x1: x2]
                image_transformed = transform(image=image_cropped)['image']

                image_transformed = image_transformed[np.newaxis, ...]
                image_transformed = image_transformed.to(device)

                pred = model(image_transformed)
                pred = get_aligned_bbox(pred, coeffs)

                score = get_iou_score(pred, label_transformed, device).mean().item()

                scores.append(score)
                tqdm_bar.update(1)

    score = np.mean(scores)
    print(f'\nScore: {score:.3f}.')
