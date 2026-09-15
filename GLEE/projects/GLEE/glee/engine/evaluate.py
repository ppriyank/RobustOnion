import json 
import os 
import torch 
import copy 
import os.path as osp
import itertools
from collections import defaultdict, OrderedDict

from detectron2.evaluation.coco_evaluation import COCOEvaluator, instances_to_coco_json
from detectron2.config import CfgNode
import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from pycocotools.cocoeval import COCOeval


# pycocotools.cocoeval.COCOeval
def process_results(coco_eval, weather_image_ids=None, Category="Weathers", report_overall = True):
    # metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    # OFFICEAL NAME  ^^^ 
    keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    Results_weather = {}
    
    coco_eval.evaluate()
    inds = coco_eval._paramsEval.imgIds
    print(f'{Category} .... ')
    for weather in weather_image_ids:
        print(f'{Category}  : {weather} {len(weather_image_ids[weather])}')
        coco_eval._paramsEval.imgIds = weather_image_ids[weather]
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results = {key:stats[i] for i, key in enumerate(keys)}
        Results_weather[weather] = copy.deepcopy(results)

    coco_eval._paramsEval.imgIds = inds
    if report_overall:
        print('OVERALL .... ')
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results = {key:stats[i] for i, key in enumerate(keys)}
        Results_weather['Overall'] = copy.deepcopy(results)

    print("\n\n:::::: Results :")
    for key in Results_weather:
        elements = Results_weather[key]
        elements = [f"{e:<4}: {elements[e]:.6f}" for e in elements]
        # for e in elements:f"{e:<5}: {elements[e]:.6f}"
        print(f"{key:<10} ::  ", "\t\t".join(elements))
    print("\n\n")
    return coco_eval


# pycocotools.cocoeval.COCOeval
def process_bddk_results(coco_eval, weather_image_ids=None, time_image_ids=None, report_overall = True):
    # metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    # OFFICEAL NAME  ^^^ 
    keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    Results_weather = {}
    Results_Time = {}
    
    coco_eval.evaluate()
    inds = coco_eval._paramsEval.imgIds
    print('Weathers .... ')
    for weather in weather_image_ids:
        print(f'Weathers  : {weather} {len(weather_image_ids[weather])}')
        coco_eval._paramsEval.imgIds = weather_image_ids[weather]
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results = {key:stats[i] for i, key in enumerate(keys)}
        Results_weather[weather] = copy.deepcopy(results)

    print('Time .... ')
    for weather in time_image_ids:
        print(f'Time  : {weather} {len(time_image_ids[weather])}')
        coco_eval._paramsEval.imgIds = time_image_ids[weather]
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results = {key:stats[i] for i, key in enumerate(keys)}
        Results_Time[weather] = copy.deepcopy(results)

    coco_eval._paramsEval.imgIds = inds
    
    print('OVERALL .... ')
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    results = {key:stats[i] for i, key in enumerate(keys)}
    Results_Time['Overall'] = copy.deepcopy(results)

    print("\n\n:::::: Results :")
    for key in Results_weather:
        elements = Results_weather[key]
        elements = [f"{e:<4}: {elements[e]:.6f}" for e in elements]
        # for e in elements:f"{e:<5}: {elements[e]:.6f}"
        print(f"{key:<15} ::  ", "\t\t".join(elements))
    
    print("\n")
    for key in Results_Time:
        elements = Results_Time[key]
        elements = [f"{e:<4}: {elements[e]:.6f}" for e in elements]
        # for e in elements:f"{e:<5}: {elements[e]:.6f}"
        print(f"{key:<15} ::  ", "\t\t".join(elements))
    print("\n\n")
    return coco_eval


def _evaluate_predictions_on_ext( coco_gt, coco_results,
    iou_type, use_fast_impl=True, max_dets_per_image=None,
    use_bddk=None, weather_image_ids=None, time_image_ids=None, city_image_ids=None, ):
    
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)

    # For COCO, the default max_dets_per_image is [1, 10, 100].
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    else:
        assert ( len(max_dets_per_image) >= 3 ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)

    coco_eval.params.maxDets = max_dets_per_image

    if use_bddk:
        coco_eval = process_bddk_results(coco_eval, weather_image_ids=weather_image_ids, time_image_ids=time_image_ids)
    else:
        coco_eval = process_results(coco_eval, weather_image_ids=weather_image_ids, Category="Weathers", report_overall = True)

    # weather_image_ids
    return coco_eval




class External_eval(COCOEvaluator):
    
    def __init__(self, dataset_name, tasks=None, **kwargs):
        super().__init__(dataset_name=dataset_name, tasks=tasks, **kwargs)
        
        # self._metadata
        # self._coco_api
        self.use_bddk = False 
        if 'bddk' in dataset_name:
            self.use_bddk = True  
            current_dir = osp.dirname(osp.abspath(__file__))
            
            root = osp.dirname(osp.dirname(osp.dirname(current_dir)))
            root = osp.dirname(osp.dirname(root))
            weather_json = osp.join(root, 'DATASET/bdd100k_labels_images_val_ANNO.json')
            with open(weather_json) as f:
                self.test_weather_labels = json.load(f)
        
        self.time_image_ids = defaultdict(list)
        self.city_image_ids = defaultdict(list)
        self.weather_image_ids = defaultdict(list)
        self._output_dir = self._output_dir + "-" + self.dataset_name
        
        
    
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            prediction = {"image_id": image_id}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, image_id)
            
            if len(prediction) > 1:
                self._predictions.append(prediction)

            if self.use_bddk:
                file_name = input['file_name'].split("/")[-1]
                ann_details = self.test_weather_labels[file_name]
                self.weather_image_ids[ ann_details['weather'] ].append( image_id ) 
                self.time_image_ids[ ann_details['timeofday'] ].append( image_id ) 
                self.city_image_ids[ ann_details['scene'] ].append( image_id ) 
            else:
                ann_details = self._coco_api.imgs[image_id]
                assert ann_details['file_name'] in input['file_name'] 
                assert input['width'] == ann_details['width']
                assert input['height'] == ann_details['height']
                weather = ann_details.get('weather', None)
                self.weather_image_ids[ weather ].append( image_id ) 

    def _eval_predictions(self, predictions, img_ids=None):
        
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)
        if self.force_tasks is not None:
            tasks = self.force_tasks
        
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            if "refcoco" in self.dataset_name:
                file_path = os.path.join(self._output_dir, "{}_instances_results.json".format(self.dataset_name))
            else:
                file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            
            coco_eval = ( _evaluate_predictions_on_ext(self._coco_api, coco_results, task, use_fast_impl=self._use_fast_impl, max_dets_per_image=self._max_dets_per_image, 
                use_bddk=self.use_bddk, weather_image_ids=self.weather_image_ids,  time_image_ids=self.time_image_ids, city_image_ids= self.city_image_ids) )

            res = self._derive_coco_results( coco_eval, task, class_names=self._metadata.get("thing_classes") )
            self._results[task] = res
        
