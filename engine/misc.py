import torch
import numpy as np

def get_predicted_bbox(bbox, coeff):
    bbox = bbox.cpu().detach().numpy()
    c_x, c_y, w, h = np.hsplit(bbox, bbox.shape[-1])

    w = w * coeff
    h = h * coeff

    x1, y1 = c_x - w / 2.0, c_y - h / 2.0
    x2, y2 = c_x + w / 2.0, c_y + h / 2.0

    x1[x1 < 0.0] = 0.0
    y1[y1 < 0.0] = 0.0

    x2[x2 > 1.0] = 1.0
    y2[y2 > 1.0] = 1.0

    bbox = np.concatenate([x1, y1, x2, y2], axis=-1)
    return bbox

def get_image_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox

    x1 = (x1 * width).round().astype(int)
    y1 = (y1 * height).round().astype(int)

    x2 = (x2 * width).round().astype(int)
    y2 = (y2 * height).round().astype(int)

    x_c = x1 / width
    y_c = y1 / height

    w_c = (x2 - x1) / width
    h_c = (y2 - y1) / height

    coeffs = torch.tensor([x_c, y_c, w_c, h_c])
    bbox = [x1, y1, x2, y2]
    return bbox, coeffs

def get_aligned_bbox(bbox, coeffs):
    x_center, y_center, width, height = torch.hsplit(bbox, bbox.shape[-1])
    x_coeff, y_coeff, w_coeff, h_coeff = torch.hsplit(coeffs, coeffs.shape[-1])

    x_center = x_center * w_coeff + x_coeff
    y_center = y_center * h_coeff + y_coeff

    width = width * w_coeff
    height = height * h_coeff

    x1, y1 = x_center - width / 2.0, y_center - height / 2.0
    x2, y2 = x_center + width / 2.0, y_center + height / 2.0

    x1, y1 = torch.clamp(x1, 0.0, 1.0), torch.clamp(y1, 0.0, 1.0)
    x2, y2 = torch.clamp(x2, 0.0, 1.0), torch.clamp(y2, 0.0, 1.0)

    x_center = x1 + (x2 - x1) / 2.0
    y_center = y1 + (y2 - y1) / 2.0

    width = x2 - x1
    height = y2 - y1

    bbox = torch.concatenate([x_center, y_center, width, height], axis=-1)
    return bbox

def get_opencv_bbox(c_x, c_y, w, h, width, height):
    x = c_x - w / 2.0
    y = c_y - h / 2.0

    x = int(round(x * width))
    y = int(round(y * height))

    w = int(round(w * width))
    h = int(round(h * height))
    return x, y, w, h
