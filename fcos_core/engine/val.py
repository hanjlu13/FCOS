# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from fcos_core.data.datasets.evaluation import evaluate
from fcos_core.utils.metric_logger import MetricLogger
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    meters = MetricLogger(delimiter="  ")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output, losses = model(images, targets)
            losses_reduced = sum(loss for loss in losses.values())
            meters.update(loss=losses_reduced)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        if i > 10:
            break
    return results_dict, meters.loss.global_avg


def eval_model(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    inference_timer=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if is_main_process():
        logger = logging.getLogger("fcos_core.eval")
    dataset = data_loader.dataset
    if is_main_process():
        logger.info("Start evaluation on ({} images).".format(len(dataset)))
    # total_timer.tic()
    predictions, losses = compute_on_dataset(
        model, data_loader, device, inference_timer
    )
    # wait for all processes to complete before measuring the time
    # synchronize()
    # total_time = total_timer.toc()
    # total_time_str = get_time_str(total_time)
    # logger.info(
    # "Total run time: {} ({} s / img per device, on {} devices)".format(
    # total_time_str, total_time * num_devices / len(dataset), num_devices
    # )
    # )
    # total_infer_time = get_time_str(inference_timer.total_time)
    # logger.info(
    # "Model inference time: {} ({} s / img per device, on {} devices)".format(
    # total_infer_time,
    # inference_timer.total_time * num_devices / len(dataset),
    # num_devices,
    # )
    # )

    # predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    return predictions, torch.tensor(losses).to(device)
