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

from ..coco.coco_eval import check_expected_results, COCOResults, summarize_per_category



def do_coco_bdd_evaluation(
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
    weather_json = os.path.join(root, 'DATASET/bdd100k_labels_images_val_ANNO.json')
    with open(weather_json) as f:
        test_weather_labels = json.load(f)
    
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"], weather_image_ids, time_image_ids = prepare_for_coco_detection(predictions, dataset, test_weather_labels)
        
    
    # coco_results["bbox"][0]

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            if dataset.coco:
                res = evaluate_predictions_on_coco(
                    dataset.coco, coco_results[iou_type], file_path, iou_type, weather_image_ids=weather_image_ids, time_image_ids=time_image_ids, display_results=display_results, 
                )
                results.update(res)
            
    logger.info(results)
    print("=====", results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    return results, coco_results

def prepare_for_coco_detection(predictions, dataset, test_weather_labels):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    weather_image_ids = {'partly cloudy':[], 'snowy':[], 'rainy':[], 'foggy':[], 'overcast':[], 'undefined':[], 'clear':[]}
    time_image_ids = {'undefined':[], 'night':[], 'dawn/dusk':[], 'daytime':[]}


    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        filename = dataset.coco.imgs[original_id]["file_name"]
        weather = test_weather_labels[filename]['weather']
        scene = test_weather_labels[filename]['scene']
        daytime = test_weather_labels[filename]['timeofday']

        weather_image_ids[weather].append(original_id)
        time_image_ids[daytime].append(original_id)

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
                        'daytime': daytime, 
                    })

    return coco_results, weather_image_ids, time_image_ids


def evaluate_predictions_on_coco(
        coco_gt, coco_results, json_result_file, iou_type="bbox",
        weather_image_ids=None, time_image_ids=None, display_results=True, 
    ):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    Results_weather = {}
    Results_time = {}
    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    results = COCOResults(iou_type)

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()

    inds = coco_eval._paramsEval.imgIds
    if display_results:
        print('Weathers .... ')
        for weather in weather_image_ids:
            print(f'Weathers  : {weather} ')
            coco_eval._paramsEval.imgIds = weather_image_ids[weather]
            coco_eval.accumulate()
            coco_eval.summarize()
            summarize_per_category(coco_eval, json_result_file.replace('.json', f'-{weather}.csv'))
            results.update(coco_eval)
            Results_weather[weather] = copy.deepcopy(results.results['bbox'])
        
        print('Time .... ')
        for time_local in time_image_ids:
            print(f'Time  : {time_local} ')
            coco_eval._paramsEval.imgIds = time_image_ids[time_local]
            coco_eval.accumulate()
            coco_eval.summarize()
            summarize_per_category(coco_eval, json_result_file.replace('.json', f'-{time_local.replace("/", "-")}.csv'))
            results.update(coco_eval)
            Results_time[time_local] = copy.deepcopy(results.results['bbox'])

        print('OVERALL .... ')
    coco_eval._paramsEval.imgIds = inds
    coco_eval.accumulate()
    coco_eval.summarize()
    summarize_per_category(coco_eval, json_result_file.replace('.json', '-overall.csv'))

    if display_results:
        print("\n\n:::::: Results :\n")
        for key in Results_weather:
            elements = Results_weather[key]
            elements = [f"{e:<4}: {elements[e]:.6f}" for e in elements]
            print(f"WEATHER-{key:<15} ::  ", "\t\t".join(elements))
        print("\n")
        for key in Results_time:
            elements = Results_time[key]
            elements = [f"{e:<4}: {elements[e]:.6f}" for e in elements]
            print(f"TIME-{key:<15} ::  ", "\t\t".join(elements))
        print("\n")

    
        key = 'bbox'
        results.update(coco_eval)
        elements = [f"{e:<4}: {results.results[key][e]:.6f}" for e in results.results[key]]
        print(f"Overall-{key:<10} ::  ", "\t\t".join(elements))
        print("\n")

    # print("\n\n:::::: WEATHER : ", Results_weather)
    # print("\n\n:::::: TIME    : ", Results_time)
    
    return coco_eval





