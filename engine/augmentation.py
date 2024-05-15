import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_augmentation_transform(augment, image_size, is_train=True, value=255):
    h = int(round(augment['hue'] * value))
    s = int(round(augment['saturation'] * value))
    v = int(round(augment['value'] * value))

    train_transform = A.Compose([
        A.Blur(p=augment['image_p']),
        A.MedianBlur(p=augment['image_p']),
        A.ToGray(p=augment['image_p']),
        A.CLAHE(p=augment['image_p']),
        A.RandomSizedBBoxSafeCrop(image_size, image_size, erosion_rate=augment['bbox_p']),
        A.HorizontalFlip(p=augment['hflip']),
        A.VerticalFlip(p=augment['vflip']),
        A.HueSaturationValue(hue_shift_limit=(-h, h),
                             sat_shift_limit=(-s, s),
                             val_shift_limit=(-v, v)),
        A.Normalize(mean=augment['mean'], std=augment['std']),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo'))

    test_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=augment['mean'], std=augment['std']),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo'))

    if is_train:
        return train_transform, test_transform
    else:
        return test_transform
