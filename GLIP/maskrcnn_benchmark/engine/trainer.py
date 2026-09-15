# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, get_rank, synchronize, reduce_sum, reduce_dict
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference
import pdb

from .utils import display_box_over_img, save_img

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
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False
    ):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass
        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if cfg.SOLVER.USE_AMP:
            with autocast():
                if len(captions) > 0:
                    loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # save checkpoints for further debug if nan happens
            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     logging.error("Losses are : {}".format(loss_dict))
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #             dict_to_save,
            #             fname
            #         )


            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #         dict_to_save,
            #         fname
            #     )
                

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
            if is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False
                    )
                    if is_main_process():
                        eval_result = _result[0].results['bbox']['AP']
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                              box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
                if eval_result < previous_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    previous_best = eval_result
                    checkpointer.save("model_best", **arguments)
                print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
                if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                    if is_main_process():
                        print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                    break

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )






def process_vanilla_output(cfg, milestone_target, scheduler, logger, optimizer, loss_dict, meters, model_ema, arguments, end, data_time, iteration, max_iter, global_rank, model):
    # Adapt the weight decay: only support multiStepLR
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        if milestone_target < len(scheduler.milestones):
            next_milestone = list(scheduler.milestones)[milestone_target]
        else:
            next_milestone = float('inf')
        if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
            gamma = scheduler.gamma
            logger.info("Drop the weight decay by {}!".format(gamma))
            for param in optimizer.param_groups:
                if 'weight_decay' in param:
                    param['weight_decay'] *= gamma
            # move the target forward
            milestone_target += 1

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = reduce_loss_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    meters.update(loss=losses_reduced, **loss_dict_reduced)
    if model_ema is not None:
        model_ema.update(model)
        arguments["model_ema"] = model_ema.state_dict()

    batch_time = time.time() - end
    end = time.time()
    meters.update(time=batch_time, data=data_time)
    eta_seconds = meters.time.global_avg * (max_iter - iteration)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

    if iteration % 20 == 0 or iteration == max_iter:
        if global_rank <= 0:
            print(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "wd: {wd:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    wd=optimizer.param_groups[0]["weight_decay"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

    return end, model_ema, arguments

def evaluation_steps(model, cfg, val_data_loader, device, no_display=None, metric='AP'):
    if is_main_process():
        print("Evaluating")
    eval_result = 0.0
    model.eval()
    
    dataset_name="coco"
    model.eval()
    _result = inference(
            model,
            val_data_loader,
            dataset_name=dataset_name, iou_types=("bbox",),
            box_only=cfg.MODEL.RPN_ONLY and (cfg.MODEL.RPN_ARCHITECTURE == "RPN" or cfg.DATASETS.CLASS_AGNOSTIC), 
            device=cfg.MODEL.DEVICE, expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL, output_folder=None, cfg=cfg,
            display_results= (not no_display), 
        )
    synchronize()
    device = f"cuda:{get_rank()}"
    if is_main_process():
        eval_result = _result[0].results['bbox'][metric]
        eval_result = torch.tensor(eval_result, device=device, dtype=torch.float64)
    else:
        eval_result = torch.tensor(0.0, device=device, dtype=torch.float64)
    eval_result = reduce_sum(eval_result)
    return eval_result 


class Dummy_Scalar():
    def __init__(self, ):
        self.do_nothing = None
    
    def scale(self, losses):
        return losses

    def step(self, optimizer):
        optimizer.step()
    
    def update(self, ):
        self.do_nothing


def setup_basic_stuff(cfg, data_loader, arguments, model, checkpoint_period, scheduler=None):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()
    scaler = Dummy_Scalar() 
    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    milestone_target = 0
    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    return logger, max_iter, start_iter, model_ema, global_rank, scaler, checkpoint_period, milestone_target, end, start_training_time

def process_vanilla_input(cfg, targets, end, iteration, arguments, images, device, model, display_images=None, data_loaders_val=None):
    
    data_time = time.time() - end
    iteration = iteration + 1
    arguments["iteration"] = iteration

    if display_images is not None :
        display_box_over_img(img=images.tensors, anno=targets, prefix_name="input", mode=display_images)
        if data_loaders_val is not None:
            if type(data_loaders_val) == list:
                for batch in data_loaders_val[0]:break 
            else:
                for batch in data_loaders_val:break 
            images, targets, image_ids, *_ = batch
            images = images.to(device, severity=cfg['severity'])
            display_box_over_img(img=images.tensors, anno=targets, prefix_name=f"test-{cfg['severity']}")
        # os.system("rsync -a *.png ucf2:~/VLM-LR/glip/GLIP/")
        quit()
    images = images.to(device)
    captions = None
    try:
        targets = [target.to(device) for target in targets]
        captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
    except:
        pass

    # Freeze language backbone
    if model: 
        if hasattr(model, "module"):
            model.module.language_backbone.eval()
        else:
            model.language_backbone.eval()

    return images, captions, targets, arguments, data_time


def multi_scale_train(cfg, model, data_loader, optimizer, scheduler,
        checkpointer, device, checkpoint_period, arguments, val_data_loader=None, meters=None, display_images=None , no_display=None):
    previous_best= 0 
    logger, max_iter, start_iter, model_ema, global_rank, scaler, checkpoint_period, milestone_target, end, start_training_time = setup_basic_stuff(cfg, data_loader, arguments, model, checkpoint_period, scheduler=scheduler)
    print(f"Evaluation at every {checkpoint_period} iterations.")
    
    # max_iter = 100000 
    # checkpoint_period = 50
    # Adapt the weight decay
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        images, captions, targets, arguments, data_time = process_vanilla_input(cfg, targets, end, iteration, arguments, images, device, model, display_images=display_images, data_loaders_val=None)
        # print(images.tensors.shape)
        # display_box_over_img(img=images.tensors, anno=targets, prefix_name="input", mode=False, categories=data_loader.dataset.categories(), border_width=1, font_size=10)
        
        with autocast():
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map)
        losses = sum(loss for loss in loss_dict.values())
        if torch.isnan(losses) or torch.isinf(losses):
            logging.error("NaN encountered, ignoring")
            losses[losses != losses] = 0
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        del images, captions, targets

        end, model_ema, arguments = process_vanilla_output(cfg, milestone_target, scheduler, logger, optimizer, loss_dict, meters, model_ema, arguments, end, data_time, iteration, max_iter, global_rank, model)

        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter) and iteration != 0 :
            logger.info(f'Starting the evaluations at {iteration}')
            eval_result = evaluation_steps(model, cfg, val_data_loader, device, no_display)
            model.train()
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if is_main_process() and eval_result > previous_best:
                previous_best = eval_result
                checkpointer.save("model_best", **arguments)

        if iteration == max_iter:
            # checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format( total_time_str, total_training_time / (max_iter)))



def multi_scale_train_n_captions(cfg, model, data_loader, optimizer, scheduler,
        checkpointer, device, checkpoint_period, arguments, val_data_loader=None, meters=None, display_images=None, no_display=None):
    previous_best= 0 
    logger, max_iter, start_iter, model_ema, global_rank, scaler, checkpoint_period, milestone_target, end, start_training_time = setup_basic_stuff(cfg, data_loader, arguments, model, checkpoint_period, scheduler=scheduler)

    # Adapt the weight decay
    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        images, captions, targets, arguments, data_time = process_vanilla_input(cfg, targets, end, iteration, arguments, images, device, model, display_images=display_images, data_loaders_val=None)
        # print(images.tensors.shape, positive_map.shape, greenlight_map.shape)
        new_captions = [t.get_field("new_captions") for t in targets]
        # [HD x 2 LQ x 2] x 2 (collector)
        new_captions = new_captions[::2]
        # for e in new_captions: e
        # display_box_over_img(img=images.tensors, anno=targets, prefix_name="input", mode=display_images)
        
        with autocast():
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map, greenlight_map = greenlight_map, new_captions=new_captions)
        losses = sum(loss for loss in loss_dict.values())
        if torch.isnan(losses) or torch.isinf(losses):
            logging.error("NaN encountered, ignoring")
            losses[losses != losses] = 0
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        del images, captions, targets

        end, model_ema, arguments = process_vanilla_output(cfg, milestone_target, scheduler, logger, optimizer, loss_dict, meters, model_ema, arguments, end, data_time, iteration, max_iter, global_rank, model)

        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter) and iteration != 0 :
            eval_result = evaluation_steps(model, cfg, val_data_loader, device, no_display)
            model.train()
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if is_main_process() and eval_result > previous_best:
                previous_best = eval_result
                checkpointer.save("model_best".format(iteration), **arguments)

        if iteration == max_iter:
            # checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format( total_time_str, total_training_time / (max_iter)))




def do_train_feats(
        cfg, model, data_loader, optimizer, scheduler, checkpointer, device, checkpoint_period,
        val_data_loader=None, start_iter=0):
    from maskrcnn_benchmark.engine.inference import inference_feats
    meters = MetricLogger(delimiter="  ")
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    arguments = {}
    max_iter = len(data_loader)
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()
    scaler = GradScaler()
    global_rank = get_rank()
    checkpoint_period = 500


    iters_per_epoch = len(data_loader) // cfg.SOLVER.MAX_EPOCH
    print(f"Iter per epoch {iters_per_epoch} / {cfg.SOLVER.MAX_EPOCH} epochs" ,  )
        

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
    
    best_acc = 0 
    for iteration, (x_f, c_f, labels) in enumerate(data_loader, start_iter):
        # print(iteration)
        data_time = time.time() - end
        iteration = iteration + 1
        
        x_f = x_f.to(device)
        c_f = c_f.to(device)
        labels = labels.to(device)

        with autocast():
            loss = model(x_f, c_f, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.error("NaN encountered, ignoring")
                break 
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        
        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if num_devices > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        meters.update(loss=loss)
        if iteration % iters_per_epoch == 0:
            if global_rank == 0:
                print(meters.delimiter.join(
                    ["eta: {eta}",   "iter: {iter}",  "{meters}",  "lr: {lr:.6f}", "wd: {wd:.6f}", "max mem: {memory:.0f}",]
                    ).format(eta=eta_string, iter=iteration, meters=str(meters), lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"], memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,)
                )
        
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        if val_data_loader and ( (iteration % (checkpoint_period) == 0)) :
            if is_main_process():
                print("Evaluating")
            model.eval()
            acc = inference_feats(model, val_data_loader, device=cfg.MODEL.DEVICE,  cfg=cfg)
            model.train()
            if cfg.SOLVER.USE_AUTOSTEP:
                scheduler.step(acc)

            if acc > best_acc:
                arguments['best_acc'] = acc
                if global_rank == 0:
                    logger.info(f"*\n Saving the best model ... {iteration} : {acc }")
                    checkpointer.save("model_best", **arguments)
                best_acc = acc
            

    
    
    if is_main_process():
        print("Evaluating")
    model.eval()
    acc = inference_feats(model, val_data_loader, device=cfg.MODEL.DEVICE,  cfg=cfg)
    arguments['best_acc'] = acc
    if global_rank == 0:
        logger.info(f"*\n Saving the Final model ... {iteration} : {acc }")
        checkpointer.save("model_final", **arguments)
    

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info( "Total training time: {} ({:.4f} s / it)".format( total_time_str, total_training_time / (max_iter) ))



    
def do_train_rl(cfg, model, data_loader, optimizer, scheduler,
        checkpointer, device, checkpoint_period, arguments, val_data_loader=None, meters=None, display_images=None , no_display=None, 
        strategy_model=None, strategy_model_optimizer=None, strategy_model_scheduler=None,
        distributed=None, args=None, NOISES=None, strategy_data_loader=None, all_noise_loaders=None,
        no_noise_pred=1, results_og = None, 
        ):

    from maskrcnn_benchmark.engine.rl_based_fns import select_action, convert_actions_to_policies, cycle_dataloader, noisy_attack, \
        train_model_on_images, update_with_noisy_loss_dict, update_strategy_network, display_distribution
    from train_net_proposed import vanilla_testing_fn
    torch.autograd.set_detect_anomaly(True)
    
    
    previous_best= 0 
    logger, max_iter, start_iter, model_ema, global_rank, scaler, checkpoint_period, milestone_target, end, start_training_time = setup_basic_stuff(cfg, data_loader, arguments, model, checkpoint_period, scheduler=scheduler)
    strategy_scaler = GradScaler()
    
    print(f"Evaluation at every {checkpoint_period} iterations.")

    # rewards = AverageMeter()
    rewards = MetricLogger(delimiter="  ")
    Noise_Counts = {f'n{e}': {x: torch.tensor(0, device=device) for x in NOISES} for e in range(no_noise_pred)}
    # print(Noise_Counts)

    # len(strategy_data_loader), len(data_loader)
    strategy_iter = cycle_dataloader(strategy_data_loader)
    strategy_model_module = strategy_model
    if distributed:
        strategy_model_module = strategy_model.module
    
    global_seed = cfg.SOLVER.SEED + args.local_rank
    
    # for name,p in strategy_model.named_parameters():print(f"{name:<30} : {p.mean().item():.4f}")
    current_step=0
    reward_size = 0
    strategy_model.train() 
    strategy_model_module.saved_log_probs = []
    strategy_model_module.saved_rewards = []
    
    for category in Noise_Counts:
        Noise_Counts[category] = reduce_dict(Noise_Counts[category], average=False)
    
    if is_main_process():
        display_distribution(Noise_Counts, NOISES)
    for iteration, (strategy_input, main_input) in enumerate( zip(strategy_iter, data_loader), start_iter):
        (images, targets, idxs, positive_map, positive_map_eval, greenlight_map, seeds) = main_input
        (strategy_images, strategy_targets, strategy_idxs, strategy_positive_map, strategy_positive_map_eval, strategy_greenlight_map) = strategy_input

        new_captions = [t.get_field("new_captions") for t in targets]
        current_step+=1

        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        images, captions, targets, arguments, data_time = process_vanilla_input(cfg, targets, end, iteration, arguments, images, device, model, display_images=display_images, data_loaders_val=None)
        strategy_images, strategy_captions, strategy_targets, _, _ = process_vanilla_input(cfg, strategy_targets, end, -1, {}, strategy_images, device, None)

        # print(images.tensors.shape)
        # display_box_over_img(img=images.tensors, anno=targets, prefix_name="input", mode=False, categories=None, border_width=10, font_size=10)
        # display_box_over_img(img=strategy_images.tensors, anno=strategy_targets, prefix_name="strategy", mode=False, categories=None, border_width=2, font_size=10)
        
        #--------Training strategy model----------#
        if current_step % args.interval_num == 0 and current_step>0:
            strategy_model.train()
            model.eval()

            if not args.debug:
                # results_curr_eval = evaluation_steps(model, cfg, val_data_loader, device, no_display=True, return_all_results=True )
                results_curr_eval = evaluation_steps(model, cfg, strategy_data_loader, device, no_display=True, metric=args.startegy_metric)
                og_metric = results_og
                curr_metric = results_curr_eval
            else:
                og_metric = 0.30331629102415675
                curr_metric = 0.3033363207357513

            
            # strategy_model_module.saved_log_probs = []
            # strategy_model_module.saved_rewards = []
            
            reward = (curr_metric - og_metric) * args.reward_magnitude
            print("------", reward)
            # strategy_model_module.saved_rewards.append(reward)
            policy_loss = update_strategy_network(strategy_model, strategy_model_optimizer, strategy_model_scheduler, strategy_model_module, reward, strategy_scaler)
            meters.update(reward=reward, policy_loss=policy_loss)
            reward_size = 0 
            results_og = curr_metric

        
        #--------Training target model---------#
        # strategy_model.eval()
        model.train()
        Noise_Counts_steps = {f'n{e}': {x:0 for x in NOISES} for e in range(no_noise_pred)}
        loss_dict = train_model_on_images(model, images, targets, captions, positive_map, greenlight_map, optimizer, scaler, scheduler)

        # for some reason we overwrite saved_log_probs and saved_rewards
        # https://github.com/prakashchhipa/ASTrA/blob/main/train_simCLR_adaptive_update.py#L566C22-L566C47
        # strategy_model_module.saved_log_probs = []
        # strategy_model_module.saved_rewards = []
        
        reward_size += len(strategy_images.tensors)
        actions = select_action(strategy_model, strategy_images.tensors, current_step, global_seed, strategy_model_module)
        policies = convert_actions_to_policies(actions, Noise_Counts, Noise_Counts_steps, NOISES, no_noise_pred=no_noise_pred)
        
        # External Image : len(strategy_images.tensors) --> N  
        # strategy_model --> produces P (no_noise_pred) predictions per N stragey images  
        # Target Images : N image genertes P variants 
        noisy_inputs = noisy_attack(all_noise_loaders, policies, batch_size = len(strategy_images.tensors), seed=seeds, indices=idxs , divisibility=cfg.DATALOADER.SIZE_DIVISIBILITY, no_noise_pred=no_noise_pred, device=device)
        
        # display_box_over_img(img=images.tensors, anno=targets, prefix_name="input", mode=False, categories=None, border_width=10, font_size=10)
        # display_box_over_img(img=noisy_inputs['n0'].tensors, anno=targets, prefix_name="n0", mode=False, categories=None, border_width=10, font_size=10)
        # display_box_over_img(img=noisy_inputs['n1'].tensors, anno=targets, prefix_name="n1", mode=False, categories=None, border_width=10, font_size=10)
        # display_box_over_img(img=noisy_inputs['n2'].tensors, anno=targets, prefix_name="n2", mode=False, categories=None, border_width=10, font_size=10)

        for e in range(no_noise_pred):
            name = f'n{e}'
            loss_dict_local = train_model_on_images(model, noisy_inputs[name], targets, captions, positive_map, greenlight_map, optimizer, scaler, scheduler)
            loss_dict = update_with_noisy_loss_dict(loss_dict, loss_dict_local)

        
        # contrastive_loss_clean = nt_xent(features_clean)
        # contrastive_loss_adv = nt_xent(features_adv)
        # similarity_loss = nt_xent(torch.stack((features_clean, features_adv), dim=1).reshape(-1, 512))
        # loss = (contrastive_loss_clean + contrastive_loss_adv) / 2 + args.sim_weight * similarity_loss
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

        # losses.update(loss.item(), inputs.size(0))
        # clean_loss.update(contrastive_loss_clean.item(),inputs.size(0))
        # adv_loss.update(contrastive_loss_adv.item(),inputs.size(0))
        # sim_loss.update(similarity_loss.item(),inputs.size(0))

        del images, captions, targets, noisy_inputs, loss_dict_local
        end, model_ema, arguments = process_vanilla_output(cfg, milestone_target, scheduler, logger, optimizer, loss_dict, meters, model_ema, arguments, end, data_time, iteration, max_iter, global_rank, model)

        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter) and iteration != 0 :
            logger.info(f'Starting the evaluations at {iteration}')
            eval_result = evaluation_steps(model, cfg, val_data_loader, device, no_display)
            model.train()
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if is_main_process() and eval_result > previous_best:
                previous_best = eval_result
                checkpointer.save("model_best", **arguments)


            for category in Noise_Counts:
                Noise_Counts[category] = reduce_dict(Noise_Counts[category], average=False)
            if is_main_process():
                display_distribution(Noise_Counts, NOISES)
        if iteration == max_iter:
            # checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format( total_time_str, total_training_time / (max_iter)))




