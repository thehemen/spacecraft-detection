# Spacecraft Detection

The task of [Pose Bowl: Detection Track](https://www.drivendata.org/competitions/260/) is to identify the boundaries of generic spacecraft in photos.

## Main features

This project applies the training of the pre-trained MobileNetV3 classifier.

The main features of this work:
- mean-std stats of all the dataset,
- unfreeze layers during the first half of the training stage,
- lower down the AdamW learning rate during the second one,
- evaluation by the GIoU score.

# Result

Public score: 0.7771.

Private score: 0.7716.
