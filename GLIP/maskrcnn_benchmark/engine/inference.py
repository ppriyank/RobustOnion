# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import re
import sys

import copy 
import torch
import torch.nn.functional as F 
from tqdm import tqdm
from collections import defaultdict

from maskrcnn_benchmark.data.datasets.evaluation import evaluate, im_detect_bbox_aug
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
import pdb
from maskrcnn_benchmark.data.datasets.evaluation.flickr.flickr_eval import FlickrEvaluator
from maskrcnn_benchmark.structures.bounding_box import BoxList
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
import numpy as np 
from .utils import save_img, normalize, display_box_over_img
from PIL import Image
import pickle
import pandas as pd 
from einops import rearrange, repeat


def imshow(img, file_name = "tmp.jpg"):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    #plt.figtext(0.5, 0.09, "test", wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(file_name)

def load(url_or_file_name):
    try:
        response = requests.get(url_or_file_name)
    except:
        response = None
    if response is None:
        pil_image = Image.open(url_or_file_name).convert("RGB")
    else:
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image
def inference_default(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, *_ = batch
        with torch.no_grad():
            if cfg.TEST.USE_MULTISCALE:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    predictions = results_dict
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return None

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def create_one_hot_dict(labels, no_minus_one_for_one_hot = False):
    positive_map_token_to_label = defaultdict(int)
    positive_map_label_to_token = defaultdict(int)

    for i in range(len(labels)):
        positive_map_token_to_label[i] = labels[i]
        positive_map_label_to_token[labels[i]] = i

    if no_minus_one_for_one_hot:
        positive_map_token_to_label = defaultdict(int)
        positive_map_label_to_token = defaultdict(int)

        for i in range(len(labels)):
            positive_map_token_to_label[i+1] = labels[i]
            positive_map_label_to_token[labels[i]] = i + 1

    return positive_map_token_to_label, positive_map_label_to_token


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = defaultdict(list)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]  # because the labels starts from 1
                positive_map_label_to_token[labels[j]].append(i)
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map, positive_map_label_to_token  # / (positive_map.sum(-1)[:, None] + 1e-6)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert(counter == len(lst))

    return all_

def create_queries_and_maps_from_dataset(dataset, cfg):
    categories = dataset.categories()
    #one_hot = dataset.one_hot

    labels = []
    label_list = []
    keys = list(categories.keys())
    keys.sort()
    for i in keys:
        labels.append(i)
        label_list.append(categories[i])

    if cfg.TEST.CHUNKED_EVALUATION != -1:
        labels = chunks(labels, cfg.TEST.CHUNKED_EVALUATION)
        label_list = chunks(label_list, cfg.TEST.CHUNKED_EVALUATION)
    else:
        labels = [labels]
        label_list = [label_list]

    all_queries = []
    all_positive_map_label_to_token = []

    for i in range(len(labels)):
        labels_i = labels[i]
        label_list_i = label_list[i]
        query_i, positive_map_label_to_token_i = create_queries_and_maps(
            labels_i, label_list_i, additional_labels = cfg.DATASETS.SUPRESS_QUERY if cfg.DATASETS.USE_SUPRESS_QUERY else None, cfg = cfg)
        
        all_queries.append(query_i)
        all_positive_map_label_to_token.append(positive_map_label_to_token_i)
    if is_main_process():
        print("All queries", all_queries)
    return all_queries, all_positive_map_label_to_token

def create_queries_and_maps(labels, label_list, additional_labels = None, cfg = None):

    # Clean label list
    original_label_list = label_list.copy()
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = cfg.DATASETS.SEPARATION_TOKENS
    
    caption_prompt = cfg.DATASETS.CAPTION_PROMPT
    if caption_prompt is not None and isinstance(caption_prompt, str):
        caption_prompt = load_from_yaml_file(caption_prompt)
    use_caption_prompt = cfg.DATASETS.USE_CAPTION_PROMPT and caption_prompt is not None
    for _index, label in enumerate(label_list):
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["prefix"]
        
        start_i = len(objects_query)

        if use_caption_prompt:
            objects_query += caption_prompt[_index]["name"]
        else:
            objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]
        
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["suffix"]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens
    
    if additional_labels is not None:
        objects_query += separation_tokens
        for _index, label in enumerate(additional_labels):
            objects_query += label
            if _index != len(additional_labels) - 1:
                objects_query += separation_tokens

    if is_main_process():
        print(objects_query)

    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenized = tokenizer(objects_query, return_tensors="pt")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        from transformers import CLIPTokenizerFast
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True, mask_token='ðŁĴĳ</w>')
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                        from_slow=True)
        tokenized = tokenizer(objects_query,
                              max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                              truncation=True,
                              return_tensors="pt")
    else:
        tokenizer = None
        raise NotImplementedError

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive,
                                                                                        labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token

def create_positive_map_label_to_token_from_positive_map(positive_map, plus = 0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token



def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def resize_box(output, targets):
    if isinstance(targets[0], dict):
        orig_target_sizes = targets[0]["orig_size"].unsqueeze(0)
    else:
        orig_target_sizes = torch.stack([targets[0].extra_fields["orig_size"] for _ in range(1)], dim=0)
    img_h, img_w = orig_target_sizes.unbind(1)
    return output.resize((img_w, img_h))

def flickr_post_process(output, targets, positive_map_label_to_token, plus):
    output = resize_box(output, targets)
    scores, indices = torch.topk(output.extra_fields["scores"], k = len(output.extra_fields["scores"]), sorted=True)
    boxes = output.bbox.tolist()
    boxes = [boxes[i] for i in indices]
    labels = [output.extra_fields["labels"][i] for i in indices]
    output_boxes = [[] for i in range(len(positive_map_label_to_token))]
    output_scores = [[] for i in range(len(positive_map_label_to_token))]
    for i in range(len(boxes)):
        output_boxes[labels[i] - plus].append(boxes[i])
        output_scores[labels[i] - plus].append(scores[i])
    for i in output_boxes:
        i.append([0.0, 0.0, 0.0, 0.0])
    image_ids = [t.extra_fields["original_img_id"] for t in targets]
    sentence_ids = [t.extra_fields["sentence_id"] for t in targets]

    return {"image_id": image_ids[0], "sentence_id": sentence_ids[0], "boxes": output_boxes, "scores": output_scores}

def build_flickr_evaluator(cfg):
    root = "DATASET/flickr30k/flickr30k/" # Hard written!!
    derived_root = cfg.DATASETS.PROFILE[0].split('flickr30k')[0]
    root = root.replace("DATASET/", derived_root)
    evaluator = FlickrEvaluator(
        root,
        subset="test" if "test" in cfg.DATASETS.TEST[0]  else "val",
        merge_boxes=cfg.DATASETS.FLICKR_GT_TYPE == "merged")
    return evaluator

def build_lvis_evaluator(ann_file, fixed_ap=True):
    from maskrcnn_benchmark.data.datasets.evaluation.lvis.lvis import LVIS
    from maskrcnn_benchmark.data.datasets.evaluation.lvis.lvis_eval import LvisEvaluatorFixedAP, LvisEvaluator
    evaluator = LvisEvaluatorFixedAP(LVIS(ann_file), fixed_ap=fixed_ap)
    #evaluator = LvisEvaluator(LVIS(ann_file), iou_types=['segm', 'bbox'])
    return evaluator

def write_lvis_results(results, output_file_name):
    lines = []
    lines.append("metric, avg ")
    for each_result in results:
        metric_string = " ".join(each_result.split(" ")[:-2])
        number = each_result.split(" ")[-1]
        each_result = metric_string + ", " + number + " "
        lines.append(each_result)

    string_to_write = "\n".join(lines) + "\n"
    with open(output_file_name, "w") as f:
        f.write(string_to_write)
    return

def write_flickr_results(results, output_file_name):
    '''
    {'Recall@1_all': 0.8394651146677753, 'Recall@1_animals': 0.9177820267686424, 'Recall@1_bodyparts': 0.7097966728280961, ...}
    '''
    lines = []
    lines.append("metric, avg ")
    for each_metric, number in results.items():
        each_result = each_metric + ", " + str(number) + " "
        lines.append(each_result)

    string_to_write = "\n".join(lines) + "\n"
    with open(output_file_name, "w") as f:
        f.write(string_to_write)
    return

def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        cfg=None,
        verbose=True,
        visualizer = None,
        display_results=True, 
    ):
    # convert to a torch.device for efficiency
    try:
        device = torch.device(device)
    except:
        device = device
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    if verbose:
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()

    task = cfg.TEST.EVAL_TASK

    if not task:
        return inference_default(model, data_loader, dataset_name, iou_types, box_only, device, expected_results, expected_results_sigma_tol, output_folder, cfg)
        
    if cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
        assert task == 'detection'
        categories = dataset.categories()

        keys = list(categories.keys())
        keys.sort()
        all_queries = [[categories[k] for k in keys]]
        all_positive_map_label_to_token = [{k: [i] for i, k in enumerate(keys)}]
    elif task == "detection":
        all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(dataset, cfg)
    elif task == "grounding":
        all_queries = [None]
        all_positive_map_label_to_token = [None]
    else:
        assert(0)

    '''
    Build Dataset Sepecific Evaluator
    '''
    if "flickr" in cfg.DATASETS.TEST[0]:
        evaluator = build_flickr_evaluator(cfg)
    elif "lvis" in cfg.DATASETS.TEST[0]:
        evaluator = build_lvis_evaluator(dataset.ann_file, fixed_ap=not cfg.DATASETS.LVIS_USE_NORMAL_AP)
    else:
        evaluator = None

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    if verbose:
        _iterator = tqdm(data_loader)
    else:
        _iterator = data_loader
    
    
    
    for i, batch in enumerate(_iterator):
        if i == cfg.TEST.SUBSET:
            break
        images, targets, image_ids, *_ = batch

        # save_img(images.tensors)
        all_output = []
        mdetr_style_output = []
        with torch.no_grad():
            if cfg.TEST.USE_MULTISCALE:
                query_time = len(all_queries)
                for query_i in range(query_time):
                    if task == "detection":
                        captions = [all_queries[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token = all_positive_map_label_to_token[query_i]
                    else:
                        captions = None
                        positive_map_label_to_token = None

                output = im_detect_bbox_aug(model, images, device, captions, positive_map_label_to_token)
                output = [o.to(cpu_device) for o in output]
                all_output.append(output)
            else:
                images = images.to(device)
                query_time = len(all_queries)

                for query_i in range(query_time):
                    if not isinstance(targets[0], dict): # For LVIS dataset and datasets directly copied from MDETR
                        targets = [target.to(device) for target in targets]
                    '''
                    different datasets seem to have different data format... For LVIS dataset, the target is a dictionary, while for modulatedDataset such as COCO/Flickr, the target is a BoxList
                    '''

                    if task == "detection":
                        captions = [all_queries[query_i] for ii in range(len(targets))]
                        positive_map_label_to_token = all_positive_map_label_to_token[query_i]
                    elif task == "grounding":
                        captions = [t.get_field("caption") for t in targets]
                        positive_map_eval = [t.get_field("positive_map_eval") for t in targets]
                        if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                            plus = 1
                        else:
                            plus = 0
                        assert(len(positive_map_eval) == 1) # Let's just use one image per batch
                        positive_map_eval = positive_map_eval[0]
                        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map_eval, plus=plus)
                    

                    ####### ####### ####### ####### ####### #######
                    ####### https://github.com/Lightning-AI/pytorch-lightning/issues/10308#issuecomment-957857971
                    # BUG IN OLD TORCH 
                    temp_dict = dict()
                    for key in positive_map_label_to_token:
                        temp_dict[key] = positive_map_label_to_token[key]                      
                    positive_map_label_to_token = temp_dict
                    ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######

                    # display_box_over_img(img=images.tensors, anno=targets, prefix_name="input", mode=True, categories=dataset.categories())
                    
                    output = model(images, captions=captions, positive_map=positive_map_label_to_token)
                    
        
                    output = [o.to(cpu_device) for o in output]

                    if "flickr" in cfg.DATASETS.TEST[0]:
                        output = output[0]
                        new_output = flickr_post_process(
                            output,
                            targets,
                            positive_map_label_to_token,
                            plus # This is only used in Flickr
                        )
                        mdetr_style_output.append(new_output)
                    elif "lvis" in cfg.DATASETS.TEST[0]:
                        output = output[0]
                        output = resize_box(output, targets)
                        scores = output.extra_fields["scores"]
                        labels = output.extra_fields["labels"]
                        boxes = output.bbox
                        mdetr_style_output.append((targets[0]["image_id"].item(), {"scores": scores, "labels": labels, "boxes": boxes}))
                    else:
                        all_output.append(output)
        # visualizer = True 
        if visualizer is not None:
            from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
            visualizer = GLIPDemo(cfg, min_image_size=800, confidence_threshold=0.7, show_mask_heatmaps=False, load_model=False)

            assert(len(all_output) == 1)
            if "lvis" in cfg.DATASETS.TEST[0]:
                scores = [o[1]["scores"] for o in mdetr_style_output]
                labels = [o[1]["labels"] for o in mdetr_style_output]
                boxes = [o[1]["boxes"] for o in mdetr_style_output]
                scores = torch.cat(scores, dim=0)
                labels = torch.cat(labels, dim=0)
                boxes = torch.cat(boxes, dim=0)
                visualizer_input = BoxList(boxes, output.size)
                visualizer_input.add_field("scores", scores)
                visualizer_input.add_field("labels", labels)
            else:
                visualizer_input = all_output[0][0] # single image_visualize

            image_id = dataset.ids[i]
            try:
                image_path = os.path.join(dataset.root, dataset.coco.loadImgs(image_id)[0]["file_name"])
                categories = dataset.coco.dataset["categories"]
            except:
                lvis = dataset.lvis
                img_id = dataset.ids[i]
                ann_ids = lvis.get_ann_ids(img_ids=img_id)
                target = lvis.load_anns(ann_ids)

                image_path = "DATASET/coco/" +  "/".join(dataset.lvis.load_imgs(img_id)[0]["coco_url"].split("/")[-2:])
                categories = dataset.lvis.dataset["categories"]

            image = load(image_path)
            no_background = True
            label_list = []
            for index, i in enumerate(categories):
                if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                    label_list.append(i["name"])
            visualizer.entities =  label_list
            
            threshold, alpha, color = 0.6, 0.5, 255
            text_size, text_pixel, box_pixel, text_offset = 1, 1, 3, 10
            text_offset_original = 4

            result, _ = visualizer.visualize_with_predictions( image, visualizer_input,  threshold, alpha=alpha, box_pixel=box_pixel, text_size=text_size, text_pixel=text_pixel, text_offset=text_offset, text_offset_original=text_offset_original, color=color,)
            imshow(result, "./visualize/img_{}.jpg".format(image_id))
        
        if evaluator is not None:
            evaluator.update(mdetr_style_output)
        else:
            output = [[row[_i] for row in all_output] for _i in range(len(all_output[0]))]
            for index, i in enumerate(output):
                output[index] = i[0].concate_box_list(i)

            results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})

    if evaluator is not None:
        evaluator.synchronize_between_processes()
        try:
            evaluator.accumulate()
        except:
            print("Evaluator has no accumulation, skipped...")
        score = evaluator.summarize()
        if is_main_process():
            print(score)
        import maskrcnn_benchmark.utils.mdetr_dist as dist
        if is_main_process():
            if "flickr" in cfg.DATASETS.TEST[0]:
                write_flickr_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
            elif "lvis" in cfg.DATASETS.TEST[0]:
                write_lvis_results(score, output_file_name=os.path.join(output_folder, "bbox.csv"))
        try:
            torch.distributed.barrier()
        except:
            print("Default process group is not initialized")
        return

    if evaluator is not None:
        predictions = mdetr_style_output
    else:
        predictions = results_dict
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    print("Accumulated results")
    if not is_main_process():
        return None

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    
    if (display_results == None or display_results == False):
        name = cfg.DATASETS.TRAIN[0]
        extra_args['display_results'] = False 
        # if ("bdd" in name or "dawn" in name or "wedge" in name \
        #     or 'FoggyCityscape' in name or 'kitti' in dataset_name ):
        #     extra_args['display_results'] = False 
        
        
        

    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)


def inference_re_validate(data_loader=None, iou_types=("bbox",), box_only=False, expected_results=(),
        expected_results_sigma_tol=4, output_folder=None, **kwargs):
    
    dataset = data_loader.dataset
    predictions = None 
    prediction_path = os.path.join(output_folder, "predictions.pth")
    if os.path.exists(prediction_path):
        predictions = torch.load(prediction_path)
    extra_args = dict(box_only=box_only, iou_types=iou_types, expected_results=expected_results, expected_results_sigma_tol=expected_results_sigma_tol,)
    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)



def inference_dump(
        model, data_loader, dataset_name,
        iou_types=("bbox",), box_only=False, device="cuda",
        expected_results=(), expected_results_sigma_tol=4, output_folder=None,
        cfg=None, verbose=True, visualizer = None):

    import cv2 
    if cfg.DATASETS.DATALOADER_MODE is None:
        save_path =  os.path.join(output_folder, "predictions-HD.pth")
    else:
        save_path =  os.path.join(output_folder, f"predictions-{cfg.DATASETS.DATALOADER_MODE}.pth")
    assert not os.path.exists(save_path), f"Dump path {save_path} already exists !!"

    
    try:
        device = torch.device(device)
    except:
        device = device
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    assert num_devices == 1, "Dumping with multi-GPUs"
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    start_time = time.time()

    task = cfg.TEST.EVAL_TASK

    if cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
        assert task == 'detection'
        categories = dataset.categories()

        keys = list(categories.keys())
        keys.sort()
        all_queries = [[categories[k] for k in keys]]
        all_positive_map_label_to_token = [{k: [i] for i, k in enumerate(keys)}]
    elif task == "detection":
        all_queries, all_positive_map_label_to_token = create_queries_and_maps_from_dataset(dataset, cfg)
    elif task == "grounding":
        all_queries = [None]
        all_positive_map_label_to_token = [None]
    else:
        assert(0)

    def fft(image):
        result_array = np.zeros_like(image)
        for i, channel in enumerate(image):
            fft = np.fft.fft2(channel)
            f_transform_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(f_transform_shifted) 
            magnitude_spectrum = np.log1p(magnitude_spectrum)
            result_array[i] = magnitude_spectrum
        return result_array



    evaluator = None
    model.eval()
    cpu_device = torch.device("cpu")
    if verbose:
        _iterator = tqdm(data_loader)
    else:
        _iterator = data_loader
    all_output = []
    accumulated_spectrum = 0 
    spectrum_counter = 0 
    with_normalization = True   
    if cfg.TEST.PREDICT == "GRAM_DUMP":
        # https://github.com/VectorInstitute/gram-ood-detection/blob/master/ResNet_Cifar100.ipynb
        # https://arxiv.org/abs/1912.12510
        
        Powers = [i+1 for i in range(10)]
        # Powers = [i+1 for i in range(7)]
        #  cfg.MODEL.SWINT.OUT_CHANNELS
        DIM = 256 
        # masks= torch.triu(torch.ones((DIM,DIM), dtype=torch.bool)).unsqueeze(0)
        # N = int( DIM * (DIM + 1) // 2 )
        # MINI = torch.zeros((4, len(Powers), N)).cuda() + float('inf')
        # MAXI = torch.zeros((4, len(Powers), N)).cuda() - float('inf')
        
        MINI = torch.zeros((4, len(Powers), DIM)).cuda() + float('inf')
        MAXI = torch.zeros((4, len(Powers), DIM)).cuda() - float('inf')
        
    elif cfg.TEST.PREDICT == "GRAM_NOISE_COMPARE":
        # https://arxiv.org/abs/1912.12510
        # https://github.com/VectorInstitute/gram-ood-detection/blob/master/ResNet_Cifar100.ipynb
        DIM = 256 
        Powers = [i+1 for i in range(10)]

        NOISE = cfg.DATASETS.DATALOADER_MODE
        GRAM_ROOT = "../Analysis/Feats/GRAM_MATRIX/"
        files = os.listdir(GRAM_ROOT)
        files = [e for e in files if "GRAM_COMPARE" not in e]
        
        N_NOISES = len(files)
        N_NOISES = N_NOISES - 1 ## remoe the current noise 
        
        MINI_NOISE = torch.zeros((N_NOISES, 4, len(Powers), DIM)).cuda() + float('inf')
        MAXI_NOISE = torch.zeros((N_NOISES, 4, len(Powers), DIM)).cuda() - float('inf')

        count = 0
        noise_labels = []
        for pik_file in files:
            if NOISE in pik_file:
                print(f"Ignoring ... {NOISE}")
                continue
            pik_file_path = os.path.join(GRAM_ROOT, pik_file)
            dict_noise = torch.load(pik_file_path)
            MINI_NOISE[count] = dict_noise['MINI']
            MAXI_NOISE[count] = dict_noise['MAXI']
            count +=1
            noise_labels.append(pik_file)
            
         
        deviations = [[] for i in range(N_NOISES)]
    
    elif cfg.TEST.PREDICT == "Feat_DUMP":
        # https://arxiv.org/pdf/2202.04933
        # 4 layers
        all_output = [[] for  i in range(4)]
    
    elif cfg.TEST.PREDICT == "ALL_DUMP":
        BACKBONE = []
        FUSED_FEATS = []
        BACKBONE_FPN = []
        FILE_NAME = []

    elif cfg.TEST.PREDICT == 'PRED_DUMP':
        # python -m pip install inflect nltk
        from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
        visualizer = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
            show_mask_heatmaps=False,
            load_model=False
        )
        box_pixel, text_size, text_pixel = 5, 3, 3
        text_offset, text_offset_original = 10, 10
        color = 1
        alpha, threshold = 1, 0.6

        results_dict = {}
        categories = dataset.coco.dataset["categories"]
        no_background = True
        label_list = []
        for index, i in enumerate(categories):
            if not no_background or (i["name"] != "__background__" and i['id'] != 0):
                label_list.append(i["name"])
        visualizer.entities =  label_list

    elif cfg.TEST.PREDICT == "SPATIAL_DUMP" or cfg.TEST.PREDICT == "SPATIAL_DUMP_CATEGORY" or cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE":
        if cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE":
            SIZE=800
            RESIZE=30 
        else:
            SIZE=400
        INDEX = 0 
        from maskrcnn_benchmark.data.transforms.transforms import Resize_Constant
        data_loader.dataset.transforms.transforms[0]= Resize_Constant((SIZE,SIZE))
        all_output = []
        externa_dataset =  "bdd" in dataset_name or "dawn" in dataset_name or "wedge" in dataset_name \
            or 'FoggyCityscape' in dataset_name or 'kitti' in dataset_name 
        if (not cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE") and (not externa_dataset):
            cfg.defrost()
            cfg.TEST.DUMP_INDEX = [1]
            cfg.freeze()
    elif cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE_NO_BICU" :
        INDEX = 0 
        all_output = []

    
    start_index, end_index = None, None 
    if cfg.TEST.DUMP_INDEX:
        if len(cfg.TEST.DUMP_INDEX) != 1:
            start_index, end_index = cfg.TEST.DUMP_INDEX

    for i, batch in enumerate(_iterator):
        # image_name = dataset.coco.loadImgs(dataset.ids[i])[0]["file_name"]
        # image_path = os.path.join(dataset.root, image_name)

        if cfg.TEST.DUMP_INDEX:
            if len(cfg.TEST.DUMP_INDEX) == 1:
                if i % 2 ==0 :continue
            elif (i < start_index or i > end_index):
                continue

        images, targets, image_ids, *_ = batch
        # save_img(images.tensors)
        local_output = []
        with torch.no_grad():
            
            query_time = len(all_queries)

            for query_i in range(query_time):
                if not isinstance(targets[0], dict): # For LVIS dataset and datasets directly copied from MDETR
                    targets = [target.to(device) for target in targets]
                
                if task == "detection":
                    captions = [all_queries[query_i] for ii in range(len(targets))]
                    positive_map_label_to_token = all_positive_map_label_to_token[query_i]
                elif task == "grounding":
                    captions = [t.get_field("caption") for t in targets]
                    positive_map_eval = [t.get_field("positive_map_eval") for t in targets]
                    if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
                        plus = 1
                    else:
                        plus = 0
                    assert(len(positive_map_eval) == 1) # Let's just use one image per batch
                    positive_map_eval = positive_map_eval[0]
                    positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map_eval, plus=plus)
                

                ####### ####### ####### ####### ####### #######
                ####### https://github.com/Lightning-AI/pytorch-lightning/issues/10308#issuecomment-957857971
                # BUG IN OLD TORCH 
                temp_dict = dict()
                for key in positive_map_label_to_token:
                    temp_dict[key] = positive_map_label_to_token[key]                      
                positive_map_label_to_token = temp_dict
                ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######

                if cfg.TEST.PREDICT == "GRAM_DUMP":
                    images = images.to(device)
                    feats = model(images, captions=captions, positive_map=positive_map_label_to_token, mean=False, true_backbone=False )
                    for i,layer in enumerate(feats[1:]):
                        layer = layer.detach()
                        layer =  F.normalize(layer, p=2.0, dim=1)
                        for j,power in enumerate(Powers):

                            res = layer ** power
                            res = ((torch.matmul(res,res.transpose(dim0=2,dim1=1)))).sum(dim=2) 
                            res = (res.sign()*torch.abs(res)**(1/power)).reshape(res.shape[0],-1)

                            current_min = res.min(dim=0)[0]
                            current_max = res.max(dim=0)[0]

                            # res = torch.bmm(res, res.permute(0,2,1))
                            # res = res ** (1 / power)
                            # res = res[masks].flatten()

                            MINI[i][j] = torch.minimum(res, MINI[i][j])
                            MAXI[i][j] = torch.maximum(res, MAXI[i][j])

                            # if (MINI[i][j] == float('inf')).sum() != 0 or (MAXI[i][j] == -float('inf')).sum() != 0:                                
                            #     import pdb
                            #     pdb.set_trace()
                            # if float('inf') in current_min or -float('inf') in current_max:
                            #     import pdb
                            #     pdb.set_trace()
                            
                elif cfg.TEST.PREDICT == "GRAM_NOISE_COMPARE": 
                    images = images.to(device)
                    feats = model(images, captions=captions, positive_map=positive_map_label_to_token, mean=False, true_backbone=False )
                    batch_deviations = [[] for i in range(N_NOISES)]
                    for i,layer in enumerate(feats[1:]):
                        layer = layer.detach()
                        layer =  F.normalize(layer, p=2.0, dim=1)
                        dev = [0 for i in range(N_NOISES)]
                        for j,power in enumerate(Powers):
                            # (layer ** 2).sum(1)
                            res = layer ** power
                            res = ((torch.matmul(res,res.transpose(dim0=2,dim1=1)))).sum(dim=2) 
                            res = (res.sign()*torch.abs(res)**(1/power)).reshape(res.shape[0],-1)

                            for noise in range(N_NOISES):
                                mins = MINI_NOISE[noise][i][j]
                                maxs = MAXI_NOISE[noise][i][j]

                                dev[noise] +=  (F.relu(  mins  - res )/ torch.abs( mins + 10**-6)).sum(dim=1,keepdim=True)
                                dev[noise] +=  (F.relu( res - maxs )/torch.abs( maxs + 10**-6 )).sum(dim=1,keepdim=True)

                        for noise in range(N_NOISES):
                            batch_deviations[noise].append( dev[noise].cpu().detach().numpy())

                        
                    for noise in range(N_NOISES):
                        batch_deviations[noise] = np.concatenate(batch_deviations[noise],axis=1)
                        deviations[noise].append(batch_deviations[noise])
                
                elif cfg.TEST.PREDICT == "Feat_DUMP": 
                    images = images.to(device)
                    feats = model(images, captions=captions, positive_map=positive_map_label_to_token, mean=True, true_backbone=True, only_last=False )

                    for i,layer in enumerate(feats):
                        if with_normalization:
                            layer =  F.normalize(layer, p=2.0, dim=-1)
                        all_output[i].append(layer.cpu().detach())
                    
                elif cfg.TEST.PREDICT == "ALL_DUMP":
                    images = images.to(device)
                    feats_dump = model(images, captions=captions, positive_map=positive_map_label_to_token)
                    
                    # [e.shape for e in feats_dump['backbone']]
                    BACKBONE.append(feats_dump['backbone'])
                    FUSED_FEATS.append(feats_dump['fused_visual_features'])
                    BACKBONE_FPN.append(feats_dump['backbone_fpn'])
                    
                elif cfg.TEST.PREDICT == 'PRED_DUMP':
                    images = images.to(device)
                    output = model(images, captions=captions, positive_map=positive_map_label_to_token)
                    output = [o.to(cpu_device) for o in output]
                    local_output.append(output)
                
                elif cfg.TEST.PREDICT == "SPATIAL_DUMP" or cfg.TEST.PREDICT == "SPATIAL_DUMP_CATEGORY" or cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE" or cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE_NO_BICU":
                    images = images.to(device)
                    # print(images.tensors.shape)
                    if cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE_NO_BICU":
                        feats = model(images, captions=captions, positive_map=positive_map_label_to_token, mean=True, true_backbone=True, only_last=False )
                    else:
                        feats = model(images, captions=captions, positive_map=positive_map_label_to_token, mean=False, true_backbone=True, only_last=False )
                    layer = feats[INDEX]
                    if cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE" :
                        HW = layer.shape[-1]
                        H = int(HW ** 0.5)
                        layer = rearrange(layer, "B C (H W) -> B C H W", H=H,  W=H)
                        layer = F.interpolate( layer, size=(RESIZE , RESIZE), mode="bilinear", align_corners=False)
                        layer = rearrange(layer, "B C H W -> B C (H W) ")
                    if with_normalization:
                        layer =  F.normalize(layer, p=2.0, dim=1)
                    all_output.append(layer.cpu().detach())
                    # for i,layer in enumerate(feats):
               
               
                        
                    

                elif cfg.TEST.PREDICT == 'True':
                    images = images.to(device)
                    feats = model(images, captions=captions, positive_map=positive_map_label_to_token)
                    feats = feats.to(cpu_device)
                    all_output.append(feats)
                else:
                    image = images.tensors.squeeze()
                    
                    image = image.numpy()
                    image = normalize(image)
                    image = fft(image)
                    
                    image = (image - image.min()) / (image.max() - image.min())
                    image *= 255.0 
                    image = cv2.resize(image.transpose(1,2,0), (800, 800))

                    accumulated_spectrum += image
                    spectrum_counter +=1

        if cfg.TEST.PREDICT == 'PRED_DUMP':
            assert(len(local_output) == 1)
            visualizer_input = local_output[0][0] # single image_visualize
            image_id = dataset.ids[i]
            
            # image_path = os.path.join(dataset.root, dataset.coco.loadImgs(image_id)[0]["file_name"])
            # image = load(image_path)
            # (426, 640, 3)
            image = normalize(images.tensors)
            # save_img(image, "temp.png")
            image = image.squeeze().permute(1,2,0).cpu().numpy()
            
            result, _ = visualizer.visualize_with_predictions( image, visualizer_input, threshold, alpha=alpha, box_pixel=box_pixel, text_size=text_size, text_pixel=text_pixel, text_offset=text_offset, text_offset_original=text_offset_original, color=color,)
            save_img(torch.tensor(result).permute(2,0,1).unsqueeze(0), f"{output_folder}/img_{image_id}.jpg")
            # imshow(result, f"{output_folder}/img_{image_id}.jpg")
            

            
            

    if cfg.TEST.PREDICT == "GRAM_DUMP":
        save_path = "GRAM_" + dataset_name + "_" + cfg.DATASETS.DATALOADER_MODE  + ".pth"
        synchronize()
        torch.save(dict(MINI=MINI.cpu(), MAXI=MAXI.cpu()), save_path)
    elif cfg.TEST.PREDICT == "GRAM_NOISE_COMPARE":
        for noise in range(N_NOISES):
            deviations[noise] = np.concatenate(deviations[noise],axis=0)
        
        save_path = "GRAM_COMPARE_" + dataset_name + "_" + cfg.DATASETS.DATALOADER_MODE  + ".pth"
        
        torch.save(
            dict( deviations=deviations,  noise_labels=noise_labels), 
        save_path)
    
    elif cfg.TEST.PREDICT == "Feat_DUMP": 
        for i,layer in enumerate(all_output):
            all_output[i] = torch.cat(layer)

        if "bdd" in dataset_name or "dawn" in dataset_name or "wedge" in dataset_name \
            or 'FoggyCityscape' in dataset_name or 'kitti' in dataset_name :
            save_path = "ALL_BACKBONE_" + dataset_name  + ".pth"
            file_names = []
            for image_id, _ in enumerate(all_output[0]):
                original_id = dataset.id_to_img_map[image_id]
                # if len(prediction) == 0:
                #     continue
                filename = dataset.coco.imgs[original_id]["file_name"]
                file_names.append(filename)
            
            if not with_normalization:
                save_path += "_NO_NORM"
            torch.save({'all_feats':all_output, 'file_names': file_names}, save_path)
        else:
            save_path = "ALL_BACKBONE_" + dataset_name + "_" + cfg.DATASETS.DATALOADER_MODE  + ".pth"
            if not with_normalization:
                save_path += "_NO_NORM"
            if cfg.MODEL.BACKBONE.CONV_BODY == "SWINT-FPN-RETINANET-SPARSE":
                save_path += "_SPARSE"
            torch.save( dict( all_feats=all_output ),  save_path)
    
    elif cfg.TEST.PREDICT == "ALL_DUMP":
        FUSED_FEATS = np.concatenate(FUSED_FEATS)
        BACKBONE_FPN = np.concatenate(BACKBONE_FPN)
        
        BACKBONE_LAYERS = [[] for i in range(4)]
        for i,layer in enumerate(BACKBONE):
            BACKBONE_LAYERS[0].append(layer[0])
            BACKBONE_LAYERS[1].append(layer[1])
            BACKBONE_LAYERS[2].append(layer[2])
            BACKBONE_LAYERS[3].append(layer[3])

        BACKBONE_LAYERS[0] = np.concatenate(BACKBONE_LAYERS[0])
        BACKBONE_LAYERS[1] = np.concatenate(BACKBONE_LAYERS[1])
        BACKBONE_LAYERS[2] = np.concatenate(BACKBONE_LAYERS[2])
        BACKBONE_LAYERS[3] = np.concatenate(BACKBONE_LAYERS[3])
        # [e.shape for e in BACKBONE_LAYERS]

        to_be_dumped = {
        "BACKBONE":  BACKBONE_LAYERS, 
        "BACKBONE_FPN" : BACKBONE_FPN, 
        "FUSED_FEATS": FUSED_FEATS, 
        }
        # pickle_name = os.path.join(output_folder, f"predictions_{start_index}_{end_index}-{sev}")
        pickle_name = os.path.join(output_folder, f"predictions_{start_index}_{end_index}")
        with open(f'{pickle_name}.pkl', 'wb') as handle:
            pickle.dump(to_be_dumped, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    elif cfg.TEST.PREDICT == "SPATIAL_DUMP" or cfg.TEST.PREDICT == "SPATIAL_DUMP_CATEGORY" or cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE" \
        or cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE_NO_BICU":
        layer = all_output
        all_output = torch.cat(layer)
        print(all_output.shape)
        # image_path = os.path.join(dataset.root, image_name)
        if cfg.TEST.PREDICT == "SPATIAL_DUMP_CATEGORY":
            import json 
            image_names = [dataset.coco.loadImgs(dataset.ids[i])[0]["file_name"] for i in range(len(_iterator))]
            if "bdd" in dataset_name:
                weather_json = os.path.join('../DATASET/bdd100k_labels_images_train_ANNO.json')
                train_weather_labels=  None 
                with open(weather_json) as f:
                    train_weather_labels = json.load(f)
                # train_weather_labels[ image_names[0] ]['weather']
                categories = [train_weather_labels[e]['weather'] for e in image_names]
                
            elif "dawn" in dataset_name:
                categories = [e.split("/")[0] for e in image_names]
            # for e in set(categories):e, (np.array(categories) == e).sum()                
            
        if "bdd" in dataset_name or "dawn" in dataset_name or "wedge" in dataset_name \
            or 'FoggyCityscape' in dataset_name or 'kitti' in dataset_name :
            save_path = "ALL_BACKBONE_" + dataset_name  + ".pth"
            file_names = []
            
            if not with_normalization:
                save_path += "_NO_NORM"
            if cfg.TEST.PREDICT == "SPATIAL_DUMP_CATEGORY":
                torch.save({'all_feats':all_output, 'categories': categories}, save_path)
            else:
                torch.save({'all_feats':all_output}, save_path)
        else:
            if cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE" or cfg.TEST.PREDICT == "SPATIAL_DUMP_RESIZE_NO_BICU":
                model_pretrained = cfg.MODEL.WEIGHT.split("/")[1]
                save_path = model_pretrained + "_SPATIAL_" + "coco" + "_" + cfg.DATASETS.DATALOADER_MODE  + ".pth"
                save_path = os.path.join(cfg.OUTPUT_DIR, save_path)
            else:
                if "coco" in dataset_name:
                    save_path = "ALL_BACKBONE_SPATIAL_" + "coco" + "_" + cfg.DATASETS.DATALOADER_MODE  + ".pth"
                else:
                    save_path = "ALL_BACKBONE_SPATIAL_" + dataset_name + "_" + cfg.DATASETS.DATALOADER_MODE  + ".pth"

            if not with_normalization:
                save_path += "_NO_NORM"
            if cfg.MODEL.BACKBONE.CONV_BODY == "SWINT-FPN-RETINANET-SPARSE":
                save_path += "_SPARSE"
            torch.save( dict( all_feats=all_output ),  save_path)
    
    


    
    elif dataset_name == 'bdd100k_val':     
        synchronize()
        all_output = torch.cat(all_output)
        
        file_names = []
        for image_id, prediction in enumerate(all_output):
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue
            filename = dataset.coco.imgs[original_id]["file_name"]
            file_names.append(filename)

        torch.save({'feat':all_output, 'file_names': file_names}, save_path)

    elif cfg.TEST.PREDICT == 'True':
        # wait for all processes to complete before measuring the time
        synchronize()
        all_output = torch.cat(all_output)
        torch.save(all_output, save_path)
    elif cfg.TEST.PREDICT == 'PRED_DUMP':
        _ = 0 
    else:
        average_spectrum = accumulated_spectrum / spectrum_counter

        average_spectrum = average_spectrum.astype(np.uint8)
        result_image = Image.fromarray(average_spectrum)
        result_image.save(save_path.replace(".pth", ".png"))
        
        
                
    return None 




def inference_feats(model, data_loader, device="cuda", cfg=None, display_results=True, verbose=True, eval_factor=5, heatmap=False, heatmap_name= None, externa_dataset=None, scores_on_eval=True):
    from scipy.stats import pearsonr, spearmanr, kendalltau
    from maskrcnn_benchmark.utils.comm import get_world_size
    from maskrcnn_benchmark.utils.comm import synchronize, get_rank
    import torch.distributed as dist 
    
    try:
        device = torch.device(device)
    except:
        device = device
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset

    all_labels = {v:k for k,v in dataset.index_to_noise.items()}
    if verbose:
        logger.info("Start evaluation on dataset({} features).".format(len(dataset)))
    start_time = time.time()

    model.eval()
    data_loader.dataset.eval = True
    results_dict = {}
    cpu_device = torch.device("cpu")
    if verbose:
        _iterator = tqdm(data_loader)
    else:
        _iterator = data_loader
    
    
    rank = get_rank()

    if externa_dataset:
        template_columns = []
        if cfg.DATALOADER.CATEGORIES:
            ext_noises = dataset.ext_noises
            template_columns = ext_noises
            ext_noises = {e:i for i,e in enumerate(ext_noises) }
            # Train on row  evalaute on column
            template = torch.zeros( (len(all_labels)), (len(dataset.ext_noises)) ).cuda(rank)
            # Train on columns ||  evalaute on rows
            gt = dataset.local_ext_df
            
        else:
            external_dataset_list = dataset.ext_data_list
            template_columns = external_dataset_list
            # Train on row  evalaute on rows
            template = torch.zeros( ((len(all_labels), len(external_dataset_list))) ).cuda(rank)
            # Train on columns  evalaute on rows
            gt = dataset.local_ext_df
    else:
        # Train on row || evalaute on rows
        template = torch.zeros( (len(all_labels)), (len(all_labels)) ).cuda(rank)
        # Train on columns ||  evalaute on rows
        gt = dataset.df
        
    
    all_output = {}
    for i, batch in enumerate(_iterator):
        x_f, y_f, labels, names = batch
        with torch.no_grad():
            x_f = x_f.to(device)
            y_f = y_f.to(device)
            labels = labels.to(device)

            output = model(x_f, y_f)
            
            if externa_dataset:
                if cfg.DATALOADER.CATEGORIES:
                    x_name, y_name = names
                    # train on row  || evaluate on column 
                    row, column = all_labels[ x_name[0] ] , ext_noises[ y_name[0] ] 
                    template[row, column] = output 
                else:
                    x_name, y_name = names[0], names[1:]
                    # train on row  || evaluate on columns
                    row = all_labels[ x_name[0] ]
                    template[row, :] = output
            else:
                x_name, y_name = names
                # train on row  || evaluate on column 
                row, column = all_labels[ x_name[0] ] , all_labels[ y_name[0] ] 
                template[row, column] = output 

    if num_devices > 1:
        # print(template)
        dist.all_reduce(template, op=dist.ReduceOp.SUM)

    # print(template)
    if externa_dataset:
        test_set = list(gt.index)
        template = pd.DataFrame(template.cpu().numpy(), columns=template_columns, index=all_labels.keys())
    else:
        test_set = all_labels.keys()
        template = pd.DataFrame(template.cpu().numpy(), columns=all_labels.keys(), index=all_labels.keys())
    
    if not scores_on_eval:
        # evaluate on rows and train on columns
        template = template.T
    else:
        # train on rows and evaluate on columns
        template = template
        gt = gt.T

    data_loader.dataset.eval = False 
    correlations = {'pearsons': [],  'spearman': [], 'name': [], 'score':[]}
    
    for name_noie in test_set :
        y = gt[name_noie]
        x = template[name_noie]

        assert (x.keys() == y.keys()).all()
        
        gt_top5 = set(y.nlargest(eval_factor ).keys())
        pred = set(x.nlargest(eval_factor ).keys())
        pred_acc = len ( pred.intersection(gt_top5)) / eval_factor

        pearson_corr, _ = pearsonr( x, y)
        spearman_corr, _ = spearmanr( x, y)
        tau, _ = kendalltau( x, y)

        correlations['pearsons'].append(pearson_corr)
        correlations['spearman'].append(spearman_corr)
        # correlations['kendalltau'].append(spearman_corr)
        correlations['name'].append(name_noie)
        correlations['score'].append(pred_acc)
    

    df_corr = pd.DataFrame(correlations)
    df_corr = df_corr.set_index('name')
    if is_main_process():
        logger.info(df_corr)
        
    if heatmap and is_main_process():    
        new_columns = {
            'pixel_dropout': "PIX Dr.", 'iso_blur': 'ISO',  'salt_pepper': "S & P", 
            'low-res': 'Pixel.', 'focus_blur': 'Focus', 'jpg_compression': 'JPG',
            'motion_blur2': 'Motion', 'fog': 'Fog', 
            "rain_model2" : 'Rain', "snow_model2": 'Snow', "atmospheric" : 'Turb.'
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname( os.path.dirname(current_dir) ) )
        
        plot_path = os.path.join(parent_dir , "Analysis", "plot")
        if plot_path not in sys.path:sys.path.append(plot_path)
        plot_path = os.path.join(parent_dir , "Analysis", )
        if plot_path not in sys.path:sys.path.append(plot_path)
        
        from plot.misc_plot import heatmap_plt
        from plot.colors import DARK_RED, lighten_color, generate_color_gradients2

        plot_config = dict(figsize=(10, 6), grid_alpha=0.5, fmt=".1f", y_label_rotate=90, 
            Y_label_fontsize=10, X_label_fontsize=12, x_label_rotate=20, cbarpos=[0.81, 0.2, 2, 0.7],
            x_label_dist=None, y_label_dist=None, color_label_fontsize=15, 
            xticklabels= 1, annot=True, yticklabels=True, ann_size=15, grid_color='black', grid_width=1 , 
        )
        
        colors = [ (0, DARK_RED ), (0.35, 'white' ) , (0.65, 'white' ), (1, lighten_color("#4169E1", 0.8) )]
        cmap = generate_color_gradients2(1000, colors)
        weight_folder = os.path.dirname(cfg.MODEL.WEIGHT)

        if scores_on_eval:
            ### train on rows and evaluate on columns
            # display evaluation as column 
            template = template.T.rename(columns=new_columns).T
            heatmap_plt(template, name=f"{weight_folder}/{heatmap_name}", vmin = template.min().min(), vmax = template.max().max(), cmap=cmap, color_bar_labels=None, color_bar_labels_range=None, **plot_config)
            heatmap_plt(df_corr, name=f"{weight_folder}/{heatmap_name}-Pred", vmin = -1, vmax = 1, cmap=cmap, color_bar_labels=None, color_bar_labels_range=None, **plot_config)
        else:
            heatmap_plt(template, name=f"{weight_folder}/{heatmap_name}", vmin = template.min().min(), vmax = template.max().max(), cmap=cmap, color_bar_labels=None, color_bar_labels_range=None, **plot_config)
            heatmap_plt(df_corr, name=f"{weight_folder}/{heatmap_name}-Pred", vmin = -1, vmax = 1, cmap=cmap, color_bar_labels=None, color_bar_labels_range=None, **plot_config)

    return df_corr.score.mean()




