import torch
from torch import nn


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (
            target_top + target_bottom
        )
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return torch.matmul(weight, losses).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class GIOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (
            target_top + target_bottom
        )
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        # calc IOU
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # find areaC
        areaC_left = torch.max(pred_left, target_left)
        areaC_top = torch.max(pred_top, target_top)
        areaC_right = torch.max(pred_right, target_right)
        areaC_bottom = torch.max(pred_bottom, target_bottom)

        # calc area of C
        areaC = (areaC_left + areaC_right) * (areaC_top + areaC_bottom)

        # calc IOU

        # cal GIOU
        iou = (area_intersect + 1.0) / (area_union + 1.0)
        giou = iou - (areaC - area_union + 1.0) / (areaC + 1.0)

        loss_giou = 1.0 - giou

        if weight is not None and weight.sum() > 0:
            return torch.matmul(weight, loss_giou).sum() / weight.sum()
        else:
            return loss_giou.mean()


# bounded IOU Loss adapted from mmdection
class Bounded_IOULoss(nn.Module):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    """

    def forward(self, pred, target, beta=0.2, eps=1e-3, weight=None):
        """
        
        Args:
            pred (tensor): Predicted bboxes.
            target (tensor): Target bboxes.
            beta (float): beta parameter in smoothl1.
            eps (float): eps to avoid NaN.
            weight ([type], optional): [description]. Defaults to None.
        
        Returns:
            loss (torch.tensor): bounded iou loss
        """
        pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
        pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
        pred_w = pred[:, 0] + pred[:, 2] + 1
        pred_h = pred[:, 1] + pred[:, 3] + 1
        with torch.no_grad():
            target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
            target_ctry = (target[:, 1] + target[:, 3]) * 0.5
            target_w = target[:, 0] + target[:, 2] + 1
            target_h = target[:, 1] + target[:, 3] + 1

        dx = target_ctrx - pred_ctrx
        dy = target_ctry - pred_ctry

        loss_dx = 1 - torch.max(
            (target_w - 2 * dx.abs()) / (target_w + 2 * dx.abs() + eps),
            torch.zeros_like(dx),
        )
        loss_dy = 1 - torch.max(
            (target_h - 2 * dy.abs()) / (target_h + 2 * dy.abs() + eps),
            torch.zeros_like(dy),
        )
        loss_dw = 1 - torch.min(
            target_w / (pred_w + eps), pred_w / (target_w + eps)
        )
        loss_dh = 1 - torch.min(
            target_h / (pred_h + eps), pred_h / (target_h + eps)
        )
        loss_comb = torch.stack(
            [loss_dx, loss_dy, loss_dw, loss_dh], dim=-1
        ).view(loss_dx.size(0), -1)

        loss = torch.where(
            loss_comb < beta,
            0.5 * loss_comb * loss_comb / beta,
            loss_comb - 0.5 * beta,
        )

        if weight is not None and weight.sum() > 0:
            return torch.matmul(weight, loss).sum() / weight.sum()
        else:
            return loss.mean()


def make_iou_loss(cfg):
    iou_loss_weights = cfg.MODEL.FCOS.IOU_LOSS_WEIGHT
    if iou_loss_weights == (1.0,):
        iou_loss_weights = None
    else:
        iou_loss_weights = torch.tensor(iou_loss_weights, dtype=torch.float32)

    extra_args = dict(weight=iou_loss_weights)
    if cfg.MODEL.FCOS.REG_LOSS_TYPE == "IOU":
        loss_calc = IOULoss()
        return loss_calc, extra_args
    elif cfg.MODEL.FCOS.REG_LOSS_TYPE == "GIOU":
        loss_calc = GIOULoss()
        return loss_calc, extra_args
    elif cfg.MODEL.FCOS.REG_LOSS_TYPE == "BIOU":
        loss_calc = Bounded_IOULoss()
        extra_args.update(
            beta=cfg.MODEL.FCOS.BIOU_BETA, eps=cfg.MODEL.FCOS.BIOU_EPS
        )
        return loss_calc, extra_args

