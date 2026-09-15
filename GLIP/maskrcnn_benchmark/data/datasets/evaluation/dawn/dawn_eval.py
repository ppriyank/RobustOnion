import logging
import tempfile
import os
import torch
import numpy as np
import json

from collections import OrderedDict
from tqdm import tqdm
import copy 

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data import datasets

from ..coco.coco_eval import check_expected_results, COCOResults, summarize_per_category
from collections import defaultdict


def do_coco_dawn_wedge_evaluation(
        dataset,
        predictions,
        output_folder,
        box_only=False,
        iou_types=("bbox",),
        expected_results=(),
        expected_results_sigma_tol=4,
        display_results=True, 
    ):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    
    if box_only:
        assert False, " not  yet verified, copy code from coco eval?"
    logger.info("Preparing results for COCO format")
    coco_results = {}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    maskrcnn_benchmark_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    root = os.path.dirname(os.path.dirname(maskrcnn_benchmark_path))

    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        if isinstance(dataset, datasets.VirtualKitti):
            coco_results["bbox"], weather_image_ids = prepare_for_coco_detection(predictions, dataset, weather_label='condition')
            spacing= 20
        elif isinstance(dataset, datasets.WIDER_FACE):
            coco_results["bbox"], weather_image_ids = prepare_for_coco_detection_no_weather(predictions, dataset)
            spacing= 20
        else:
            coco_results["bbox"], weather_image_ids = prepare_for_coco_detection(predictions, dataset)
            spacing= 10
    
    # coco_results["bbox"][0]
    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            if dataset.coco:
                res = evaluate_predictions_on_coco(dataset.coco, coco_results[iou_type], file_path, iou_type, weather_image_ids=weather_image_ids, display_results=display_results, spacing=spacing)
                results.update(res)
    
    logger.info(results)
    print("=====", results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results


def do_coco_wedge_evaluation(
        dataset,
        predictions,
        output_folder,
        box_only=False,
        iou_types=("bbox",),
        expected_results=(),
        expected_results_sigma_tol=4,
        display_results=True, 
    ):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        assert False, " not  yet verified, copy code from coco eval?"
    logger.info("Preparing results for COCO format")
    coco_results = {}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    maskrcnn_benchmark_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    root = os.path.dirname(os.path.dirname(maskrcnn_benchmark_path))

    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        if isinstance(dataset, datasets.VirtualKitty):
            coco_results["bbox"], weather_image_ids = prepare_for_coco_detection(predictions, dataset, weather_label='condition')
        else:
            coco_results["bbox"], weather_image_ids = prepare_for_coco_detection(predictions, dataset)
    
    # coco_results["bbox"][0]
    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            if dataset.coco:
                res = evaluate_predictions_on_coco(dataset.coco, coco_results[iou_type], file_path, iou_type, weather_image_ids=weather_image_ids, display_results=display_results)
                results.update(res)
    
    logger.info(results)
    print("=====", results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results



def prepare_for_coco_detection(predictions, dataset, weather_label="weather"):
    
    coco_results = []
    weather_image_ids = defaultdict(list)
    
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        filename = dataset.coco.imgs[original_id]["file_name"]
        weather = dataset.coco.imgs[original_id][weather_label]
        
        weather_image_ids[weather].append(original_id)
        
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        for k, box in enumerate(boxes):
            if labels[k] in dataset.contiguous_category_id_to_json_id:
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": dataset.contiguous_category_id_to_json_id[labels[k]],
                        "bbox": box,
                        "score": scores[k],
                        "weather": weather, 
                    })

    
    # add Ground Truth as predictions 
    # for i in dataset.id_to_img_map:
    #     original_id = dataset.id_to_img_map[i]
    #     weather = dataset.coco.imgs[original_id][weather_label]
    #     ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
    #     target = dataset.coco.loadAnns(ann_ids)
    #     for box in target:
    #         coco_results.append(
    #             {
    #                 "image_id": original_id,
    #                 "category_id": box['category_id'],
    #                 "bbox": box['bbox'],
    #                 "score": 1,
    #                 "weather": weather, 
    #             })
                
    return coco_results, weather_image_ids


def prepare_for_coco_detection_no_weather(predictions, dataset, weather_label="weather"):
    
    coco_results = []
    weather_image_ids = None 
    
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        filename = dataset.coco.imgs[original_id]["file_name"]
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        for k, box in enumerate(boxes):
            if labels[k] in dataset.contiguous_category_id_to_json_id:
                coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": dataset.contiguous_category_id_to_json_id[labels[k]],
                        "bbox": box,
                        "score": scores[k],
                    })
            
    return coco_results, weather_image_ids



def evaluate_predictions_on_coco(
        coco_gt, coco_results, json_result_file, iou_type="bbox",
        weather_image_ids=None,  display_results=True, spacing = 10, 
    ):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    Results_weather = {}
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    results = COCOResults(iou_type)

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    
    inds = coco_eval._paramsEval.imgIds
    
    if display_results and weather_image_ids is not None :
        print('Weathers .... ')
        for weather in weather_image_ids:
            print(f'Weathers  : {weather} {len(weather_image_ids[weather])}')
            coco_eval._paramsEval.imgIds = weather_image_ids[weather]
            coco_eval.accumulate()
            coco_eval.summarize()
            summarize_per_category(coco_eval, json_result_file.replace('.json', f'-{weather}.csv'))
            results.update(coco_eval)
            Results_weather[weather] = copy.deepcopy(results.results['bbox'])
    
        print('OVERALL .... ')
    
    coco_eval._paramsEval.imgIds = inds
    coco_eval.accumulate()
    coco_eval.summarize()
    summarize_per_category(coco_eval, json_result_file.replace('.json', '-overall.csv'))
    results.update(coco_eval)
    Results_weather['Overall'] = copy.deepcopy(results.results['bbox'])

    if display_results and weather_image_ids is not None :
        print("\n\n:::::: Results :")
        for key in Results_weather:
            elements = Results_weather[key]
            elements = [f"{e:<4}: {elements[e]:.6f}" for e in elements]
            # for e in elements:f"{e:<5}: {elements[e]:.6f}"
            print(f"{key:<{spacing}} ::  ", "\t\t".join(elements))
        print("\n\n")

    return coco_eval





