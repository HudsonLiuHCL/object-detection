"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from torchvision.ops import nms as nms_torch

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]

@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`.
    """
    x_center = locations[:, 0]
    y_center = locations[:, 1]
    
    x1 = gt_boxes[:, 0]
    y1 = gt_boxes[:, 1]
    x2 = gt_boxes[:, 2]
    y2 = gt_boxes[:, 3]
    
    left = x_center - x1
    top = y_center - y1
    right = x2 - x_center
    bottom = y2 - y_center
    
    deltas = torch.stack([left, top, right, bottom], dim=1)

    deltas = deltas / stride
    
    background_mask = (gt_boxes[:, 0] < 0)
    deltas[background_mask, :] = -1
    
    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:

    deltas_unnorm = deltas * stride

    deltas_unnorm = torch.clamp(deltas_unnorm, min=0)
    
    x_center = locations[:, 0]
    y_center = locations[:, 1]

    left = deltas_unnorm[:, 0]
    top = deltas_unnorm[:, 1]
    right = deltas_unnorm[:, 2]
    bottom = deltas_unnorm[:, 3]

    x1 = x_center - left
    y1 = y_center - top
    x2 = x_center + right
    y2 = y_center + bottom
    
    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    return output_boxes



def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Compute centerness targets from LTRB deltas.
    Centerness = sqrt((min(l,r) * min(t,b)) / (max(l,r) * max(t,b)))
    """
    left = deltas[:, 0]
    top = deltas[:, 1]
    right = deltas[:, 2]
    bottom = deltas[:, 3]
    
    lr_min = torch.min(left, right)
    lr_max = torch.max(left, right)
    tb_min = torch.min(top, bottom)
    tb_max = torch.max(top, bottom)
    
    centerness = torch.sqrt((lr_min * tb_min) / (lr_max * tb_max + 1e-8))
    
    background_mask = (deltas[:, 0] < 0)
    centerness[background_mask] = -1
    
    return centerness


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image.
    """
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        
        _, _, H, W = feat_shape
        

        shift_x = (torch.arange(0, W, dtype=dtype, device=device) + 0.5) * level_stride
        shift_y = (torch.arange(0, H, dtype=dtype, device=device) + 0.5) * level_stride
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location_coords[level_name] = torch.stack([shift_x, shift_y], dim=1)
    
    return location_coords

def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    # Use torchvision NMS.
    keep = nms_torch(boxes_for_nms, scores, iou_threshold)
    return keep
