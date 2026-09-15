from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .vg import vg_evaluation
from .box_aug import im_detect_bbox_aug
from .od_to_grounding import od_to_grounding_evaluation

from .bdd100k import do_coco_bdd_evaluation
from .dawn import do_coco_dawn_wedge_evaluation

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.DAWN) or isinstance(dataset, datasets.WEDGE) or isinstance(dataset, datasets.VirtualKitti) or isinstance(dataset, datasets.WIDER_FACE):
        return do_coco_dawn_wedge_evaluation(**args)
    elif isinstance(dataset, datasets.BDD100K):
        return do_coco_bdd_evaluation(**args)
    elif isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset) or isinstance(dataset, datasets.BDD100K_Grounding):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))


def evaluate_mdetr(dataset, predictions, output_folder, cfg):
   
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset) or isinstance(dataset, datasets.TSVDataset):
        return coco_evaluation(**args)
    # elif isinstance(dataset, datasets.VGTSVDataset):
    #     return vg_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    elif isinstance(dataset, datasets.CocoDetectionTSV):
        return od_to_grounding_evaluation(**args)
    elif isinstance(dataset, datasets.LvisDetection):
        pass
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
