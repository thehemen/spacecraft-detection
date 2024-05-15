import cv2
import yaml
import torch
import argparse
import numpy as np
from timeit import default_timer as timer
from engine.model import SpaceshipDetector
from engine.logger import Logger
from engine.dataset import read_label
from engine.metric import get_iou_score
from engine.augmentation import get_augmentation_transform
from engine.misc import get_predicted_bbox
from engine.misc import get_image_bbox
from engine.misc import get_aligned_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect a spaceship on the image.')
    parser.add_argument('--dataset_cfg', default='spaceship.yaml')
    parser.add_argument('--augment_cfg', default='augment.yaml')
    parser.add_argument('--model_name', default='model.pt')
    parser.add_argument('--image_name', default='images/0a0ba7d4c31cd8c12e66f3e792a9599f.jpg')
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--bbox_coeff', type=float, default=5.0)
    parser.add_argument('--class_name', default='Spaceship')
    parser.add_argument('--log_path', default='runs/')
    args = parser.parse_args()

    with open(args.dataset_cfg, 'r') as f:
        dataset_cfg = yaml.load(f, Loader=yaml.CLoader)

    with open(args.augment_cfg, 'r') as f:
        augment_cfg = yaml.load(f, Loader=yaml.CLoader)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}.')

    t1 = timer()

    image = cv2.imread(args.image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    width, height = image.shape[:2][::-1]

    transform = get_augmentation_transform(augment_cfg, args.image_size, is_train=False)
    logger = Logger(log_path=args.log_path, mode_name='predict')

    label_name = args.image_name.replace('images', 'labels').replace('jpg', 'txt')
    label = read_label(label_name)

    label.append(args.class_name)

    transformed = transform(image=image, bboxes=[label])
    image_transformed = transformed['image']

    label_transformed = torch.tensor(transformed['bboxes'][0][:-1])[np.newaxis, ...]
    image_transformed = image_transformed[np.newaxis, ...]

    label_transformed = label_transformed.to(device)
    image_transformed = image_transformed.to(device)

    t2 = timer()

    model = SpaceshipDetector()
    model.load_state_dict(torch.load(args.model_name))

    model = model.to(device)
    model.eval()

    t3 = timer()

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

    t4 = timer()

    label = label[:-1]
    pred = pred.cpu().detach().numpy()[0]

    image_name = args.image_name.split('/')[1]
    logger.save_image(image_name, image, label, pred, args.class_name)

    t5 = timer()

    preprocess_time = t2 - t1
    inference_time = t4 - t3
    postprocess_time = t5 - t4
    total_time = preprocess_time + inference_time + postprocess_time

    print(f'\nPreprocess Time\tInference Time\tPostprocess Time\tTotal Time')
    print(f'{preprocess_time:.3f}\t\t{inference_time:.3f}\t\t{postprocess_time:.3f}\t\t\t{total_time:.3f}')
    print()
    print(f'Score: {score:.3f}.')
