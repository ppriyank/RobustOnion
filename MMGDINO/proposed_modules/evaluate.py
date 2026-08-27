
from mmdet.registry import METRICS
from mmdet.evaluation.metrics.coco_metric import CocoMetric

import itertools
import numpy as np 

import tempfile
import json 
import os.path as osp
import copy 
from collections import defaultdict, OrderedDict
from mmengine.logging import MMLogger
from mmengine.fileio import dump, get_local_path, load
from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from terminaltables import AsciiTable
NONE_IMPORT = ""



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


# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/coco_metric.py
@METRICS.register_module()
class Ext_evaluate(CocoMetric):

    def __init__(self, ann_file = None, sort_categories: bool = False, **kwargs):
        super().__init__(ann_file=ann_file, classwise=True, sort_categories=sort_categories, **kwargs)
        
        self.use_bddk = False 
        if 'bdd100k' in ann_file:
            self.use_bddk = True  
            current_dir = osp.dirname(osp.abspath(__file__))
            root = osp.dirname(osp.dirname(current_dir))
            weather_json = osp.join(root, 'DATASET/bdd100k_labels_images_val_ANNO.json')
            with open(weather_json) as f:
                self.test_weather_labels = json.load(f)

            self.time_image_ids = defaultdict(list)
            self.city_image_ids = defaultdict(list)

        self.weather_image_ids = defaultdict(list)
        

    def process(self, data_batch, data_samples):
        
        for i,data_sample in enumerate(data_samples):
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            
            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']

            if self.use_bddk:
                file_name = data_batch['data_samples'][i].img_path.split("/")[-1] 
                ann_details = self.test_weather_labels[file_name]
                self.weather_image_ids[ ann_details['weather'] ].append( data_sample['img_id'] ) 
                self.time_image_ids[ ann_details['timeofday'] ].append( data_sample['img_id'] ) 
                self.city_image_ids[ ann_details['scene'] ].append( data_sample['img_id'] ) 
            else:
                ann_details = self._coco_api.imgs[data_sample['img_id']]
                assert ann_details['file_name'] in data_batch['data_samples'][i].img_path 
                assert gt['width'] == ann_details['width']
                assert gt['height'] == ann_details['height']
                self.weather_image_ids[ ann_details['weather'] ].append( data_sample['img_id'] ) 

            self.results.append((gt, result))

    def compute_metrics(self, results):
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        assert self._coco_api is not None
        
        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)
        
        
        eval_results = OrderedDict()
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')
            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                coco_dt = self._coco_api.loadRes(predictions)
            except IndexError:
                logger.error('The testing results of the whole dataset is empty.')
                break

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            if self.use_bddk:
                # self.city_image_ids
                process_bddk_results(coco_eval, weather_image_ids=self.weather_image_ids, time_image_ids=self.time_image_ids, report_overall = False)
            else:
                process_results(coco_eval, weather_image_ids=self.weather_image_ids)

            if self.classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = coco_eval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, cat_id in enumerate(self.cat_ids):
                    t = []
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self._coco_api.loadCats(cat_id)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    t.append(f'{nm["name"]}')
                    t.append(f'{round(ap, 3)}')
                    eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                    # indexes of IoU  @50 and @75
                    for iou in [0, 5]:
                        precision = precisions[iou, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{round(ap, 3)}')

                    # indexes of area of small, median and large
                    for area in [1, 2, 3]:
                        precision = precisions[:, :, idx, area, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{round(ap, 3)}')
                    results_per_category.append(tuple(t))

                num_columns = len(results_per_category[0])
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [
                    'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                    'mAP_m', 'mAP_l'
                ]
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                logger.info('\n' + table.table)

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results