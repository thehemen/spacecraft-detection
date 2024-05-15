import math
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
from engine.dataset import SpaceshipDataset
from engine.model import SpaceshipDetector
from engine.fine_tune import FineTuneScheduler
from engine.cross_validation import get_cv_split
from engine.metric import get_iou_score, get_loss_score
from engine.augmentation import get_augmentation_transform
from engine.misc import get_predicted_bbox
from engine.misc import get_aligned_bbox
from engine.logger import Logger

def train_loop(dataloader, model, device, optimizer, output_step=5):
    model.train()
    loss_hist = []

    num_batches = len(dataloader)
    score = 0.0

    with tqdm(total=num_batches, position=0, leave=True) as tqdm_bar:
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            now_score = get_iou_score(pred, y, device)

            loss = get_loss_score(pred, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            score += now_score.mean().item()

            loss = loss.item()
            loss_hist.append(loss)

            tqdm_bar.update(1)

            if i % output_step == 0 or i == num_batches - 1:
                tqdm_bar.set_description(f'[{np.mean(loss_hist):.3f}]')

    loss_value = np.mean(loss_hist)
    score_value = score / num_batches
    return loss_value, score_value

def test_loop_1(dataloader, model, device, coeff, output_step=5):
    model.eval()
    preds = []

    num_batches = len(dataloader)

    with torch.no_grad():
        with tqdm(total=num_batches, position=0, leave=True) as tqdm_bar:
            for i, (X, y) in enumerate(dataloader):
                X = X.to(device)
                y = y.to(device)

                pred = model(X)
                pred = get_predicted_bbox(pred, coeff)

                preds.append(pred)
                tqdm_bar.update(1)

    preds = np.concatenate(preds)
    return preds

def test_loop_2(dataloader, model, device, output_step=5):
    model.eval()
    loss_hist = []

    num_batches = len(dataloader)
    score = 0.0

    with torch.no_grad():
        with tqdm(total=num_batches, position=0, leave=True) as tqdm_bar:
            for i, (X, y, coeffs) in enumerate(dataloader):
                X = X.to(device)
                y = y.to(device)

                coeffs = coeffs.to(device)

                pred = model(X)
                pred = get_aligned_bbox(pred, coeffs)

                loss = get_loss_score(pred, y)

                now_score = get_iou_score(pred, y, device)
                score += now_score.mean().item()

                loss = loss.item()
                loss_hist.append(loss)

                tqdm_bar.update(1)

                if i % output_step == 0 or i == num_batches - 1:
                    tqdm_bar.set_description(f'[{np.mean(loss_hist):.3f}]')

    loss_value = np.mean(loss_hist)
    score_value = score / num_batches
    return loss_value, score_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect a spaceship on the image.')
    parser.add_argument('--dataset_cfg', default='spaceship.yaml')
    parser.add_argument('--augment_cfg', default='augment.yaml')
    parser.add_argument('--lr0', type=float, default=1e-3)
    parser.add_argument('--lrf', type=float, default=1e-2)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--ft_epoch', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--bbox_coeff', type=float, default=2.0)
    parser.add_argument('--share', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--log_path', default='runs/')
    args = parser.parse_args()

    with open(args.dataset_cfg, 'r') as f:
        dataset_cfg = yaml.load(f, Loader=yaml.CLoader)

    with open(args.augment_cfg, 'r') as f:
        augment_cfg = yaml.load(f, Loader=yaml.CLoader)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}.')

    num_workers = torch.get_num_threads()
    print(f'Threads: {num_workers}.\n')

    train_transform, test_transform = get_augmentation_transform(augment_cfg, args.image_size)

    print(f'{train_transform}\n')
    print(f'{test_transform}\n')

    train_idx, test_idx = get_cv_split(args.share)

    train_dataset = SpaceshipDataset(dataset_cfg, train_idx, train_transform)
    test_dataset = SpaceshipDataset(dataset_cfg, test_idx, test_transform)

    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, num_workers=num_workers, shuffle=False)

    model = SpaceshipDetector()
    model = model.to(device)

    logger = Logger(log_path=args.log_path, config=vars(args))

    print(model)
    print(f'Log data is saved to {logger.model_path}.')

    optimizer = AdamW(model.parameters(), lr=args.lr0, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lambda x: float(x / args.warmup_steps))

    layer_num = model.get_layer_num()
    ft_epoch = int(math.floor(args.ft_epoch * args.epochs))

    ft_scheduler = FineTuneScheduler(layer_num, ft_epoch)
    ft_scheduler.init(model)

    for i in range(args.epochs):
        print(f'\nEpoch {i + 1}.')

        if i == args.warmup_steps:
            scheduler = None

        elif i == ft_epoch:
            step_num = int(math.floor(args.epochs - ft_epoch) / args.step_size)
            gamma = math.exp(math.log(args.lrf) / step_num)
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=gamma)

        train_loss, train_score = train_loop(train_dataloader, model, device, optimizer)

        test_dataset.bboxes = None
        preds = test_loop_1(test_dataloader, model, device, args.bbox_coeff)

        test_dataset.bboxes = preds
        val_loss, val_score = test_loop_2(test_dataloader, model, device)

        if ft_scheduler:
            ft_scheduler.step(i, model)

            if i == ft_epoch:
                ft_scheduler = None

        if scheduler:
            scheduler.step()

        logger.update(i, model, train_loss, train_score, val_loss, val_score)
        print(f'Score: {val_score:.3f}.')

    score = logger.best_score
    print(f'\nScore: {score:.3f}.')
