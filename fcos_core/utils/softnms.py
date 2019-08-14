import torch


def box_softnms(
    bboxes,
    scores,
    nms_threshold=0.1,
    soft_conf_thresh=0.2,
    sigma=0.5,
    mode="union",
):
    """
    soft-nms implentation according the soft-nms paper
    :param bboxes:
    :param scores:
    :param labels:
    :param nms_threshold:
    :param soft_conf_thresh:
    :return:
    """
    box_keep = []
    scores_keep = []
    weights = scores.clone()
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    orig_ids = torch.tensor(range(len(weights)))
    _, order = weights.sort(0, descending=True)
    while order.numel() > 0:
        try:
            i = order[0]
        except IndexError:
            order = order.unsqueeze(0)
            i = order[0]
        box_keep.append(orig_ids[i])

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        if mode == "union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "min":
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError("Unknown nms mode: %s." % mode)

        ids_t = (ovr >= nms_threshold).nonzero().squeeze()

        weights[[order[ids_t + 1]]] *= torch.exp(
            -(ovr[ids_t] * ovr[ids_t]) / sigma
        )

        ids = (weights[order[1:]] >= soft_conf_thresh).nonzero().squeeze()
        if ids.numel() == 0:
            break
        bboxes = bboxes[order[1:]][ids]
        weights = weights[order[1:]][ids]
        orig_ids = orig_ids[order[1:]][ids]
        if orig_ids.dim() == 0:
            orig_ids = orig_ids.unsqueeze(0)
        _, order = weights.sort(0, descending=True)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0)
            weights = weights.unsqueeze(0)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    return torch.tensor(box_keep)
