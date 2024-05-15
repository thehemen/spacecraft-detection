import os
import cv2
import glob
import torch
import shutil
import matplotlib.pyplot as plt
from engine.misc import get_opencv_bbox

def get_run_index(log_path, mode_name):
    runs = sorted(glob.glob(f'{log_path}{mode_name}*/'))

    if len(runs) > 0:
        path = runs[-1].split('/')[1]
        num = int(''.join(x for x in path if x.isdigit())) + 1
    else:
        num = 1

    return num

def plot(split_type, metric_name, model_path, values):
    fig = plt.subplots()
    plt.title(f'{split_type} {metric_name}')
    plt.plot(range(1, len(values) + 1), values, 'r')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name}')
    plt.savefig(f'{model_path}/{split_type.lower()}_{metric_name.lower()}_chart.png')
    plt.close()

class Logger:
    def __init__(self, log_path, config=None, mode_name='train'):
        self.__best_score = -1.0
        self.best_epoch = 0

        self.log_path = log_path

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        num = get_run_index(self.log_path, mode_name)

        self.model_path = f'{self.log_path}{mode_name}{num}/'
        os.mkdir(self.model_path)

        if mode_name == 'train':
            with open(f'{self.model_path}/log.txt', 'w') as f:
                f.write(f'Epoch\tTrain Loss\tTrain Score\tVal Loss\tVal Score\n')

            self.__train_loss_history = []
            self.__train_score_history = []

            self.__val_loss_history = []
            self.__val_score_history = []

    @property
    def best_score(self):
        best_score = self.__best_score if self.__best_score > 0.0 else 0.0
        return best_score

    def update(self, epoch, model, train_loss, train_score, val_loss, val_score):
        if val_score > self.__best_score:
            self.__best_score = val_score
            self.best_epoch = epoch
            torch.save(model.state_dict(), f'{self.model_path}/model.pt')

        self.__train_loss_history.append(train_loss)
        self.__train_score_history.append(train_score)

        self.__val_loss_history.append(val_loss)
        self.__val_score_history.append(val_score)

        with open(f'{self.model_path}/log.txt', 'a') as f:
            f.write(f'{epoch + 1}\t\t{train_loss:.3f}\t\t{train_score:.3f}\t\t{val_loss:.3f}\t\t{val_score:.3f}\n')

        plot('Train', 'Loss', self.model_path, self.__train_loss_history)
        plot('Train', 'Score', self.model_path, self.__train_score_history)
        plot('Val', 'Loss', self.model_path, self.__val_loss_history)
        plot('Val', 'Score', self.model_path, self.__val_score_history)

    def save_image(self, image_name, image, label, pred, class_name):
        height, width, channels = image.shape

        c_x, c_y, w, h = label

        x, y, w, h = get_opencv_bbox(c_x, c_y, w, h, width, height)
        image = cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)

        c_x, c_y, w, h = pred

        x, y, w, h = get_opencv_bbox(c_x, c_y, w, h, width, height)
        image = cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)

        cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
        cv2.imwrite(f'{self.model_path}/{image_name}', image)
