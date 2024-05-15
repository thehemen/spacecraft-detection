import torch
from math import pi

def get_iou_score(box1, box2, device):
    '''
        GIoU score.
        Pose Bowl: Spacecraft Detection and Pose Estimation Challenge.
        https://github.com/drivendataorg/spacecraft-pose-object-detection-runtime
        scripts/score.py
    '''

    zeros = torch.zeros(1,)
    zeros = zeros.to(device)

    x1_1 = (box1[:, 0] - box1[:, 2]) / 2.0
    y1_1 = (box1[:, 1] - box1[:, 3]) / 2.0

    x2_1 = (box1[:, 0] + box1[:, 2]) / 2.0
    y2_1 = (box1[:, 1] + box1[:, 3]) / 2.0

    x1_2 = (box2[:, 0] - box2[:, 2]) / 2.0
    y1_2 = (box2[:, 1] - box2[:, 3]) / 2.0

    x2_2 = (box2[:, 0] + box2[:, 2]) / 2.0
    y2_2 = (box2[:, 1] + box2[:, 3]) / 2.0

    xmin = torch.maximum(x1_1, x1_2)
    xmax = torch.minimum(x2_1, x2_2)
    ymin = torch.maximum(y1_1, y1_2)
    ymax = torch.minimum(y2_1, y2_2)

    pred_height = y2_1 - y1_1
    pred_width = x2_1 - x1_1

    box1_height = y2_2 - y1_2
    box1_width = x2_2 - x1_2

    intersection_height = torch.maximum(ymax - ymin, zeros)
    intersection_width = torch.maximum(xmax - xmin, zeros)
    area_of_intersection = intersection_height * intersection_width

    area_of_union = (
            box1_height * box1_width + pred_height * pred_width - area_of_intersection
    )

    return area_of_intersection / area_of_union

def get_loss_score(box1, box2, eps=1e-7):
    '''
        CioU score
        Ultralytics YOLOv8 object detection model.
        https://github.com/ultralytics/ultralytics
        ultralytics/utils/metrics.py
    '''

    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

    c2 = cw.pow(2) + ch.pow(2) + eps

    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4 

    v = (4 / pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)

    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    ciou = iou - (rho2 / c2 + v * alpha)
    loss = (1.0 - ciou).mean()
    return loss
