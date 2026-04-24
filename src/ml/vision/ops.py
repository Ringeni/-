import torch
from torchvision.ops import box_iou as tv_box_iou


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    return tv_box_iou(boxes1, boxes2)


def clip_boxes_to_image(boxes: torch.Tensor, size) -> torch.Tensor:
    height, width = int(size[0]), int(size[1])
    clipped = boxes.clone()
    clipped[..., 0::2] = clipped[..., 0::2].clamp(0, width)
    clipped[..., 1::2] = clipped[..., 1::2].clamp(0, height)
    return clipped


def dets_select(*args, **kwargs):
    raise NotImplementedError("dets_select is not implemented in local runtime")
