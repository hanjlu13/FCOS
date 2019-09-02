# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from fcos_core.utils.comm import (
    get_world_size,
    is_pytorch_1_1_0_or_later,
    is_main_process,
    reduce_tensor,
    synchronize,
)
from fcos_core.data.datasets.evaluation import evaluate
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.utils.tf_logger import TensorboardWriter
from .val import eval_model
from .inference import _accumulate_predictions_from_multiple_gpus
from fcos_core.solver.lookahead import Lookahead


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    val_period,
    arguments,
    use_lookahead=False,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    if use_lookahead:
        lookahead = Lookahead(optimizer, k=5, alpha=0.5)
    for iteration, (images, targets, _) in enumerate(train_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        if use_lookahead:
            lookahead.step()
        else:
            optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_hours = eta_seconds / 3600
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if is_main_process():
                TensorboardWriter.write_scalar(
                    ["train/lr", "train/mem", "train/eta"],
                    [
                        optimizer.param_groups[0]["lr"],
                        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        eta_hours,
                    ],
                    iteration,
                )
                # write losses
                TensorboardWriter.write_scalars(
                    ["train/losses", "train/time"],
                    [
                        meters.get_metric(
                            metric_names=[
                                "loss",
                                "loss_centerness",
                                "loss_cls",
                                "loss_reg",
                            ]
                        ),
                        meters.get_metric(metric_names=["time", "data"]),
                    ],
                    iteration,
                )

        if ((iteration) % val_period == 0) or (iteration == max_iter):
            predictions, val_loss = eval_model(model, val_loader)
            synchronize()
            val_loss = reduce_tensor(val_loss)
            predictions = _accumulate_predictions_from_multiple_gpus(
                predictions
            )
            if is_main_process():
                extra_args = dict(
                    box_only=False,
                    iou_types=("bbox",),
                    expected_results=cfg.VAL.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.VAL.EXPECTED_RESULTS_SIGMA_TOL,
                )
                _, _, ap_stats = evaluate(
                    dataset=val_loader.dataset,
                    predictions=predictions,
                    output_folder=None,
                    **extra_args
                )
                ap_keys = list(ap_stats.keys())
                keys = ["val/loss"] + [
                    "val/{}".format(_key) for _key in ap_keys
                ]
                vals = [float(val_loss.cpu().numpy())] + [
                    ap_stats[_key] for _key in ap_keys
                ]
                TensorboardWriter.write_scalar(keys, vals, iteration)
            synchronize()
            model.train()

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
