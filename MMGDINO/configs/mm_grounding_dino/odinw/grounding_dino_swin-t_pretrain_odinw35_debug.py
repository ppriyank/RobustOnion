_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'  # noqa

dataset_type = 'CocoDataset'
data_root = 'data/odinw/'

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = ('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'text',
                                       'custom_entities', 'caption_prompt')


# ---------------------7 brackishUnderwater---------------------#
class_name = ('crab', 'fish', 'jellyfish', 'shrimp', 'small_fish', 'starfish')
metainfo = dict(classes=class_name)
_data_root = data_root + 'brackishUnderwater/960x540/'
dataset_brackishUnderwater = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='valid/annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    pipeline=_base_.test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_brackishUnderwater = dict(
    type='CocoMetric',
    ann_file=_data_root + 'valid/annotations_without_background.json',
    metric='bbox')

# ---------------------8 ChessPieces---------------------#
class_name = ('  ', 'black bishop', 'black king', 'black knight', 'black pawn',
              'black queen', 'black rook', 'white bishop', 'white king',
              'white knight', 'white pawn', 'white queen', 'white rook')
metainfo = dict(classes=class_name)
_data_root = data_root + 'ChessPieces/Chess Pieces.v23-raw.coco/'
dataset_ChessPieces = dict(
    type=dataset_type,
    metainfo=metainfo,
    data_root=_data_root,
    ann_file='Odinw35_new_annotations/ChessPieces-new_annotations_without_background.json',
    data_prefix=dict(img='valid/'),
    pipeline=_base_.test_pipeline,
    test_mode=True,
    return_classes=True)
val_evaluator_ChessPieces = dict(
    type='CocoMetric',
    ann_file='Odinw35_new_annotations/ChessPieces-new_annotations_without_background.json',
    metric='bbox')







# --------------------- Config---------------------#

dataset_prefixes = [
    'brackishUnderwater',
    'ChessPieces',
]

datasets = [
    dataset_brackishUnderwater, 
    dataset_ChessPieces,
]

metrics = [
    val_evaluator_brackishUnderwater, val_evaluator_ChessPieces,
]

# -------------------------------------------------#
val_dataloader = dict(
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator
