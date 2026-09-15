# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.structures.image_list import to_image_list

import pdb
class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        positive_map = None
        positive_map_eval = None
        greenlight_map = None


        # images.tensor
        # target : captions (shuffled) || 
        # greenlight_map:: position of characters for correct label 
        # positive_map every box tokenized version of labels 
        
        if isinstance(targets[0], dict):
            return images, targets, img_ids, positive_map, positive_map_eval

        if "greenlight_map" in transposed_batch[1][0].fields():
            greenlight_map = torch.stack([i.get_field("greenlight_map") for i in transposed_batch[1]], dim = 0)

        if "positive_map" in transposed_batch[1][0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map").shape[1] for v in transposed_batch[1]])
            nb_boxes = sum([v.get_field("positive_map").shape[0] for v in transposed_batch[1]])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in transposed_batch[1]:
                cur_pos = v.get_field("positive_map")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            positive_map = batched_pos_map.float()
        

        if "positive_map_eval" in transposed_batch[1][0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map_eval").shape[1] for v in transposed_batch[1]])
            nb_boxes = sum([v.get_field("positive_map_eval").shape[0] for v in transposed_batch[1]])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in transposed_batch[1]:
                cur_pos = v.get_field("positive_map_eval")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
            positive_map_eval = batched_pos_map.float()


        return images, targets, img_ids, positive_map, positive_map_eval, greenlight_map


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        # return list(zip(*batch))
        transposed_batch = list(zip(*batch))

        images = transposed_batch[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        positive_map = None
        positive_map_eval = None

        if isinstance(targets[0], dict):
            return images, targets, img_ids, positive_map, positive_map_eval

        return images, targets, img_ids, positive_map, positive_map_eval



class Replica_BatchCollator(BatchCollator):
    
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        images = transposed_batch[0]
        # N_LR = images[0].shape[1]
        N_BR = images[0].shape[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        nsample = len(targets)

        new_targets = []
        new_img_ids = []
        new_set_images = []

        for i in range(nsample):
            new_targets += [targets[i]] * N_BR
            new_img_ids += [img_ids[i]] * N_BR
            for k in range(N_BR):
                new_set_images.append( images[i][k] )



        images = to_image_list(new_set_images, self.size_divisible)
        # print('^^^', self.size_divisible, len(new_set_images), type(new_set_images[0]))
        

        # images.tensors = rearrange(images.tensors, "B N C H W -> (B N) C H W ")
        targets = new_targets
        img_ids = new_img_ids
        
        positive_map = None
        positive_map_eval = None
        greenlight_map = None

        if "greenlight_map" in targets[0].fields():
            greenlight_map = torch.stack([i.get_field("greenlight_map") for i in targets], dim = 0)

        if "positive_map" in targets[0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map").shape[1] for v in targets])
            nb_boxes = sum([v.get_field("positive_map").shape[0] for v in targets])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in targets:
                cur_pos = v.get_field("positive_map")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            positive_map = batched_pos_map.float()
        

        if "positive_map_eval" in targets[0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map_eval").shape[1] for v in targets])
            nb_boxes = sum([v.get_field("positive_map_eval").shape[0] for v in targets])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in targets:
                cur_pos = v.get_field("positive_map_eval")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
            positive_map_eval = batched_pos_map.float()


        return images, targets, img_ids, positive_map, positive_map_eval, greenlight_map


class Replica_BatchCollator_RelicaEnd(BatchCollator):
    
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        images = transposed_batch[0]
        N_BR = images[0].shape[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        nsample = len(targets)

        
        new_targets = targets * N_BR
        new_img_ids = img_ids * N_BR
        new_set_images = []

        for k in range(N_BR):
            for i in range(nsample):
                new_set_images.append( images[i][k] )
                
        images = to_image_list(new_set_images, self.size_divisible)
        targets = new_targets
        img_ids = new_img_ids
        
        positive_map = None
        positive_map_eval = None
        greenlight_map = None

        if "greenlight_map" in targets[0].fields():
            greenlight_map = torch.stack([i.get_field("greenlight_map") for i in targets], dim = 0)

        if "positive_map" in targets[0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map").shape[1] for v in targets])
            nb_boxes = sum([v.get_field("positive_map").shape[0] for v in targets])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in targets:
                cur_pos = v.get_field("positive_map")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            positive_map = batched_pos_map.float()
        

        if "positive_map_eval" in targets[0].fields():
            # we batch the positive maps here
            # Since in general each batch element will have a different number of boxes,
            # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
            max_len = max([v.get_field("positive_map_eval").shape[1] for v in targets])
            nb_boxes = sum([v.get_field("positive_map_eval").shape[0] for v in targets])
            batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
            cur_count = 0
            for v in targets:
                cur_pos = v.get_field("positive_map_eval")
                batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                cur_count += len(cur_pos)

            assert cur_count == len(batched_pos_map)
            # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
            positive_map_eval = batched_pos_map.float()


        return images, targets, img_ids, positive_map, positive_map_eval, greenlight_map



class Replica_BatchCollator_Seed(Replica_BatchCollator):
    
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        images = transposed_batch[0]
        N_BR = images[0].shape[0]
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        nsample = len(targets)
        seeds = transposed_batch[3]

        images, targets, img_ids, positive_map, positive_map_eval, greenlight_map = super().__call__(batch)
        
        new_seeds = []
        for i in range(nsample):
            new_seeds += [seeds[i]] * N_BR
            
        return images, targets, img_ids, positive_map, positive_map_eval, greenlight_map, new_seeds 


