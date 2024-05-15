import glob
import numpy as np
from sklearn.model_selection import train_test_split

def get_cv_split(share, is_train=True, seed=42):
    indexes = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob('images/*.jpg'))]
    train_idx, val_idx = train_test_split(indexes, train_size=share, random_state=seed)
    train_idx = np.array([indexes.index(x) for x in train_idx])
    val_idx = np.array([indexes.index(x) for x in val_idx])

    if is_train:
        return train_idx, val_idx
    else:
        return val_idx
