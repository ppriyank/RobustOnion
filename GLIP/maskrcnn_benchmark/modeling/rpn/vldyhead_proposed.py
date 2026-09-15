import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict

from .vldyhead import VLDyHeadModule, VLDyHead
from ..utils import cat, concat_box_prediction_layers, permute_and_flatten

from maskrcnn_benchmark.utils.shallow_contrastive_loss_helper import dist 
from einops import rearrange, repeat

from maskrcnn_benchmark.utils.dist import is_dist_avail_and_initialized
import copy 

def gather_features(x, gather_fn=None, world_size=1):
    tensor_list = [torch.zeros_like(x) for _ in range(world_size)]
    x = gather_fn(tensor_list, x)
    x= torch.cat(x, dim=0)
    return x

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist



class TripletLoss(object):
    def __init__(self, margin, metric='euclidean', world_size=1):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.metric = metric
        self.collect_from_gpus = None
        self.world_size = world_size

    def enable_distirbuted(self):
        import vlkit
        self.all_gather = vlkit.ops.all_gather
        self.collect_from_gpus = True 

    def _label2similarity(sekf, label1, label2):
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar):
        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

    def __call__(self, emb1, emb2, label1, label2, normalize_feature=True):
        if self.collect_from_gpus:            
            emb1 = gather_features(emb1.contiguous(), self.all_gather, world_size=self.world_size)
            emb2 = gather_features(emb2.contiguous(), self.all_gather, world_size=self.world_size)
            label1 = gather_features(label1.contiguous(), self.all_gather, world_size=self.world_size)
            label2 = gather_features(label2.contiguous(), self.all_gather, world_size=self.world_size)
            
        # print(label1, label2)
        if normalize_feature:
            emb2 = F.normalize(emb2, p=2, dim=-1)
            emb1 = F.normalize(emb1, p=2, dim=-1)

        # print(emb2.shape, emb1.shape, label1, label2)
        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')
            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            mat_dist = euclidean_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')
            margin_label = torch.ones_like(hard_p)

        return self.ranking_loss(hard_n, hard_p, margin_label)




class VLDyHead_LANG_TRI(VLDyHead):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.N_BR = cfg.DATASETS.N_BR # number of points per sample (HD & LQ)
        self.N_LR = cfg.DATASETS.N_LR # no of caption variants per sample

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.loss = TripletLoss(margin=0.3, world_size=self.world_size)
        if is_dist_avail_and_initialized():
            self.loss.enable_distirbuted()

        print("Lang Triplet Loss VLDyHead_LANG_TRI ... ")                  
        
        self.mode = None 
        self.fusion_method = self.fuse_with_vision
        if cfg.MODEL.RPN.NO_FUSE == 'one_anchor':
            self.mode = cfg.MODEL.RPN.NO_FUSE
            self.fusion_method = self.fuse_with_one_anchor

    def fuse_with_vision(self, x, language_dict_features_new, unique_ids):
        dyhead_tower_Dirty = None 
        B = unique_ids
        N_x = len(x)
        x_variant = []
        for k in range(N_x) :
            e = repeat(x[k], "B C H W -> (B N) C H W", N=self.N_LR)
            x_variant.append(e)
        
        # x_variant[-1].mean(-1).mean(-1).mean(-1)
        # language_dict_features_new['embedded'].shape
        feat_inputs_variant = {"visual": x_variant, "lang": language_dict_features_new}
        dyhead_tower_variant = self.dyhead_tower(feat_inputs_variant)        
        return dyhead_tower_variant["visual"][-1]

    def fuse_with_one_anchor(self, x, language_dict_features_new, unique_ids):
        dyhead_tower_Dirty = None 
        B = unique_ids
        N_x = len(x)
        x_variant = []
        for k in range(N_x) :
            e = repeat(x[k][1::self.N_BR], "B C H W -> (B N) C H W", N=self.N_LR * self.N_BR)
            x_variant.append(e)
        
        # x_variant[-1].mean(-1).mean(-1).mean(-1)
        # language_dict_features_new['embedded'].shape
        # 1 1  1 1  1 1 
        feat_inputs_variant = {"visual": x_variant, "lang": language_dict_features_new}
        dyhead_tower_variant = self.dyhead_tower(feat_inputs_variant)        
        return dyhead_tower_variant["visual"][-1]

    
    def triplet_loss(self, x_variant, x, unique_ids):
        B = unique_ids
        labels_y = torch.arange(1, unique_ids + 1).cuda() + unique_ids * self.rank
        # print(labels_y)
        # print(self.world_size, self.rank)
        labels_x_1 =  repeat(labels_y, "B  -> B N", N=self.N_BR * self.N_LR ).reshape(-1)
        labels_x_2 =  repeat(labels_y, "B  -> B N", N=self.N_BR).reshape(-1)

        x_variant  =  rearrange(x_variant , "B C H W -> B C (H W)").mean(-1)
        x  =  rearrange(x , "B C H W -> B C (H W)").mean(-1)

        x = torch.cat([x_variant, x], 0)
        labels_x = torch.cat([labels_x_1, labels_x_2], 0)
        # print(x.shape, labels_x)
        triplet_loss  = self.loss(emb1=x, emb2=x, label1=labels_x, label2=labels_x, normalize_feature=True)
        return triplet_loss
        
    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None, language_dict_features_new=None ):
        logits = []
        bbox_reg = []
        centerness = []
        
        feat_inputs = {"visual": x, "lang": language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        triplet_loss = torch.tensor(0.0).cuda()
        if language_dict_features_new is not None:
            unique_ids = x[-1].shape[0] // self.N_BR
            variant_fused_embedding = self.fusion_method(x, language_dict_features_new, unique_ids=unique_ids)
            # self.fusion_method == self.fuse_with_vision
            # B x self.N_LR, C, H, W 
            # (x_0,c1, x_0,c2... || x_1,c1, x_1,c2 ... || ... )

            # self.fusion_method == self.fuse_with_one_anchor
            # B x self.N_LR x self.N_BR, C, H, W 
            # fused features of only dirty embeddings :: (x_1,c1, x_1,c2... || x_3,c1, x_3,c2 ... || ... )
            triplet_loss = self.triplet_loss(variant_fused_embedding, dyhead_tower["visual"][-1], unique_ids=unique_ids)
            del language_dict_features_new

        # soft token
        t_logits = None
        embedding = dyhead_tower["lang"]["hidden"]
        
        # MLM loss
        mlm_logits = None
        # contrastive
        contrastive_logits = None
        proj_tokens = None
            
        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        dot_product_logits = []
        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        # w/o norm
        # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 28.0)
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        fused_visual_features = None
        
        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            
            x = dyhead_tower["visual"][l]
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                fused_visual_features.append(x)
            B, C, H, W = x.shape

            # add bias (language)
            dot_product_proj_queries = self.dot_product_projection_image(x)
            dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            dot_product_logits.append(dot_product_logit)

        
        # import pdb
        # pdb.set_trace()
        
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features, triplet_loss

class VLDyHead_LANG_DISTILL(VLDyHead):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.N_BR = cfg.DATASETS.N_BR # number of points per sample (HD & LQ)
        self.N_LR = cfg.DATASETS.N_LR # no of caption variants per sample
        print(" VLDyHead_LANG_DISTILL ... ")                  

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None, language_dict_features_new=None , visual_features_HD_teacher=None):
        logits = []
        bbox_reg = []
        centerness = []
        
        feat_inputs = {"visual": x, "lang": language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        triplet_loss = torch.tensor(0.0).cuda()
        if visual_features_HD_teacher is not None:
            unique_ids = x[-1].shape[0] // self.N_BR

            language_dict_features_new = dict()
            language_dict_features_new['aggregate'] = language_dict_features['aggregate'][::2]
            language_dict_features_new['embedded'] = language_dict_features['embedded'][::2]
            language_dict_features_new['masks'] = language_dict_features['masks'][::2]
            language_dict_features_new['hidden'] = language_dict_features['hidden'][::2]
            language_dict_features_new['mlm_labels'] = language_dict_features['mlm_labels']
            
            
            feat_inputs = {"visual": visual_features_HD_teacher, "lang": language_dict_features_new}
            dyhead_tower_HD = self.dyhead_tower(feat_inputs)

            feat_teacher = repeat(dyhead_tower_HD["visual"][-1], "B C H W -> (B 2) C (H W)")
            feat_teacher = feat_teacher.mean(-1)
            feat_teacher = F.normalize(feat_teacher, p=2, dim=-1)

            feat_student = rearrange(dyhead_tower["visual"][-1], "B C H W -> B C (H W)")
            feat_student = feat_student.mean(-1)
            feat_student = F.normalize(feat_student, p=2, dim=-1)

            triplet_loss = ((feat_teacher - feat_student) ** 2).sum(-1)
            triplet_loss = triplet_loss.sum()
            
            del language_dict_features_new, dyhead_tower_HD, feat_inputs, visual_features_HD_teacher, feat_teacher, feat_student

        # soft token
        t_logits = None
        embedding = dyhead_tower["lang"]["hidden"]
        
        # MLM loss
        mlm_logits = None
        # contrastive
        contrastive_logits = None
        proj_tokens = None
            
        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        dot_product_logits = []
        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        # w/o norm
        # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 28.0)
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        fused_visual_features = None
        
        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            
            x = dyhead_tower["visual"][l]
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                fused_visual_features.append(x)
            B, C, H, W = x.shape

            # add bias (language)
            dot_product_proj_queries = self.dot_product_projection_image(x)
            dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            dot_product_logits.append(dot_product_logit)

        
        # import pdb
        # pdb.set_trace()
        
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features, triplet_loss


class VLDyHead_LANG_TRI_NOFUSE(VLDyHead_LANG_TRI):
    
    def __init__(self, mode = "replica", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.triplet_loss = self.replica_triplet_loss
        if mode == 'noise':
            self.triplet_loss = self.noise_triplet_loss
            self.dropout = nn.Dropout(p=0.1)

        if mode == 'mse':
            self.triplet_loss = self.mse_loss
        if mode == 'neg_triple':
            self.triplet_loss = self.process_neg_triplet
            self.loss = TripletLoss(margin=0.3, world_size=self.world_size)
            if is_dist_avail_and_initialized():
                self.loss.enable_distirbuted()
        
    def replica_triplet_loss(self, x, unique_ids):
        B = unique_ids
        labels_y = torch.arange(1, unique_ids + 1).cuda() + unique_ids * self.rank
        labels_x =  repeat(labels_y, "B  -> B N", N=self.N_BR).reshape(-1)
        x  =  rearrange(x , "B C H W -> B C (H W)").mean(-1)
        triplet_loss  = self.loss(emb1=x, emb2=x, label1=labels_x, label2=labels_x, normalize_feature=True)
        # print(labels_x)
        return triplet_loss
    
    def noise_triplet_loss(self, x, unique_ids):
        B = unique_ids
        
        labels_y = torch.arange(1, unique_ids + 1).cuda() + unique_ids * self.rank
        labels_x_1 =  repeat(labels_y, "B  -> B N", N=self.N_LR ).reshape(-1)
        labels_x_2 =  repeat(labels_y, "B  -> B N", N=self.N_LR * self.N_BR).reshape(-1)

        x  =  rearrange(x , "B C H W -> B C (H W)").mean(-1)
    
        x_noisy = repeat(x, "B ... -> (B N) ...", N=self.N_BR)
        # print(x.mean(-1) ,"\n", x_noisy.mean(-1))

        x_noisy += torch.randn_like(x_noisy) * (0.1**0.5)
        x_noisy = self.dropout(x_noisy)

        x = torch.cat([x_noisy, x], 0)
        labels_x = torch.cat([labels_x_2, labels_x_1], 0)
        # print(labels_x)
        triplet_loss  = self.loss(emb1=x, emb2=x, label1=labels_x, label2=labels_x, normalize_feature=True)
        
        return triplet_loss

    def mse_loss(self, x, unique_ids):
        B = unique_ids

        labels_y = torch.arange(1, unique_ids + 1).cuda() + unique_ids * self.rank
        labels_x_1 =  repeat(labels_y, "B  -> B N", N= self.N_BR ).reshape(-1)
        
        # print(labels_x_1)
        x  =  rearrange(x , "B C H W -> B C (H W)").mean(-1)
        x = F.normalize(x, p=2, dim=-1)

        mse_loss = x.unsqueeze(0)- x.unsqueeze(1)
        mse_loss = (mse_loss ** 2).sum(-1)
        # print(mse_loss)
        pairwise_label = labels_x_1.unsqueeze(0) ==  labels_x_1.unsqueeze(1)
        mse_loss = mse_loss[pairwise_label]
        mse_loss = mse_loss.sum() / 2
        return mse_loss

    def process_neg_triplet(self, x, language_dict_features, unique_ids, dyhead_tower_clean, high_start_pid=100):
        language_new_dict_features = dict()
        for key in language_dict_features.keys():
            language_new_dict_features[key] = language_dict_features[key]
            if language_new_dict_features[key] is not None:
                # language_new_dict_features[key].mean(-1)
                # tensor([-0.0109, -0.0109, -0.0112, -0.0112, -0.0110, -0.0110], device='cuda:0')
                language_new_dict_features[key] = torch.roll(language_new_dict_features[key], shifts=(self.N_BR), dims=(0))

            # ['aggregate', 'embedded', 'masks', 'hidden', 'mlm_labels']:

        feat_inputs = {"visual": x, "lang": language_new_dict_features}
        dyhead_tower_dirty = self.dyhead_tower(feat_inputs)
        x_dirty =  dyhead_tower_dirty["visual"][-1]
        x_clean =  dyhead_tower_clean["visual"][-1]
        
        x_dirty  =  rearrange(x_dirty , "B C H W -> B C (H W)").mean(-1)
        x_clean  =  rearrange(x_clean , "B C H W -> B C (H W)").mean(-1)

        del dyhead_tower_dirty, feat_inputs
        
        labels_y = torch.arange(1, unique_ids + 1).cuda() + unique_ids * self.rank        
        labels_x_1 =  repeat(labels_y, "B  -> B N", N=self.N_BR ).reshape(-1)
        labels_x_2 =  labels_x_1 + high_start_pid

        x = torch.cat([x_dirty, x_clean], 0)
        labels_x = torch.cat([labels_x_1, labels_x_2], 0)
        
        triplet_loss  = self.loss(emb1=x, emb2=x, label1=labels_x, label2=labels_x, normalize_feature=True)
        return triplet_loss
        
    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None, language_dict_features_new=None ):
        logits = []
        bbox_reg = []
        centerness = []
        
        feat_inputs = {"visual": x, "lang": language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        triplet_loss = torch.tensor(0.0).cuda()
        # print("===", self.training)
        if self.training:
            unique_ids = x[-1].shape[0] // self.N_BR
            if self.mode == 'neg_triple':
                triplet_loss =  self.process_neg_triplet(x, language_dict_features, unique_ids, dyhead_tower)
            else:
                triplet_loss = self.triplet_loss(x[-1], unique_ids=unique_ids)


        # soft token
        t_logits = None
        embedding = dyhead_tower["lang"]["hidden"]
        
        # MLM loss
        mlm_logits = None
        # contrastive
        contrastive_logits = None
        proj_tokens = None
            
        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        dot_product_logits = []
        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        # w/o norm
        # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 28.0)
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        fused_visual_features = None
        
        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            
            x = dyhead_tower["visual"][l]
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                fused_visual_features.append(x)
            B, C, H, W = x.shape

            # add bias (language)
            dot_product_proj_queries = self.dot_product_projection_image(x)
            dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            dot_product_logits.append(dot_product_logit)

        
        # import pdb
        # pdb.set_trace()
        
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features, triplet_loss



class VLDyHeadModule_LANG_TRI(VLDyHeadModule):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        del self.head
        if cfg.MODEL.RPN.NO_FUSE == "one_anchor":
            self.head = VLDyHead_LANG_TRI(cfg=cfg)        
        elif cfg.MODEL.RPN.NO_FUSE == "distill":
            self.head = VLDyHead_LANG_DISTILL(cfg=cfg)        
        elif cfg.MODEL.RPN.NO_FUSE:
            self.head = VLDyHead_LANG_TRI_NOFUSE(cfg=cfg, mode=cfg.MODEL.RPN.NO_FUSE)        
        else:
            self.head = VLDyHead_LANG_TRI(cfg)        
        print("Lang Triplet Loss VLDyHeadModule_LANG_TRI ... ")                  

    def forward(self, images, features, targets=None, language_dict_features=None,
                positive_map=None, captions=None, swint_feature_c4=None,
                language_dict_features_new = None):
        
        # return super().forward(images, features, targets=targets, language_dict_features=language_dict_features, positive_map=positive_map, captions=captions, swint_feature_c4=swint_feature_c4)
        # [e.shape for e in features]
        embedding = language_dict_features['embedded']
        text_masks = language_dict_features["masks"]

        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features, triplet_loss = self.head( 
            features, language_dict_features, embedding, swint_feature_c4, language_dict_features_new=language_dict_features_new
        )
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       proj_tokens,
                                       contrastive_logits,
                                       dot_product_logits,
                                       text_masks,
                                       mlm_logits = mlm_logits,
                                       mlm_labels = language_dict_features["mlm_labels"],
                                       shallow_img_emb_feats=shallow_img_emb_feats,
                                       fused_visual_features=fused_visual_features,
                                       triplet_loss = triplet_loss, 
                                       )
        else:
            return self._forward_test(box_regression, centerness, anchors, box_cls, token_logits, dot_product_logits, positive_map, fused_visual_features=fused_visual_features)

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors,
                       captions=None,
                       positive_map=None,
                       token_logits=None,
                       proj_tokens=None,
                       contrastive_logits=None,
                       dot_product_logits=None,
                       text_masks=None,
                       mlm_logits=None,
                       mlm_labels=None,
                       shallow_img_emb_feats=None,
                       fused_visual_features=None,
                       triplet_loss=0, 
                       ):

        loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors,
            captions,
            positive_map,
            token_logits,
            proj_tokens,
            contrastive_logits,
            dot_product_logits,
            text_masks,
            shallow_img_emb_feats
        )

        losses = {
            # "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            'triplet_loss': triplet_loss, 
        }

        if mlm_labels is not None and mlm_logits is not None:
            losses["mlm_loss"] = nn.CrossEntropyLoss(ignore_index = -100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)) * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            losses["loss_cls"] = loss_box_cls
        else:
            losses["loss_cls"] = 0.0 * loss_box_cls

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            losses["loss_token"] = loss_token * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            losses["loss_contrastive_align"] = loss_contrastive_align * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            losses["loss_dot_product_token"] = loss_dot_product_token * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS or \
                self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            losses["loss_shallow_contrastive"] = loss_shallow_contrastive * \
                                                 self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT

        if self.cfg.MODEL.RPN_ONLY:
            return None, losses, None
        else:
            # Let's just use one image per batch
            assert (box_regression[0].shape[0]) == 1
            positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)
            boxes = self.box_selector_train(box_regression, centerness, anchors,
                                        box_cls,
                                        token_logits,
                                        dot_product_logits,
                                        positive_map=positive_map_label_to_token
                                        )
            train_boxes = []
            for b, t in zip(boxes, targets):
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                train_boxes.append(cat_boxlist([b, tb]))
            return train_boxes, losses, fused_visual_features

    
class VLDyHeadModule_LANG_Distill(VLDyHeadModule_LANG_TRI):

    
    def forward(self, images, features, targets=None, language_dict_features=None,
                positive_map=None, captions=None, swint_feature_c4=None,
                language_dict_features_new = None, visual_features_HD_teacher=None):
        
        embedding = language_dict_features['embedded']
        text_masks = language_dict_features["masks"]

        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features, triplet_loss = self.head( 
            features, language_dict_features, embedding, swint_feature_c4, language_dict_features_new=language_dict_features_new,
            visual_features_HD_teacher=visual_features_HD_teacher, 
        )
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       proj_tokens,
                                       contrastive_logits,
                                       dot_product_logits,
                                       text_masks,
                                       mlm_logits = mlm_logits,
                                       mlm_labels = language_dict_features["mlm_labels"],
                                       shallow_img_emb_feats=shallow_img_emb_feats,
                                       fused_visual_features=fused_visual_features,
                                       triplet_loss = triplet_loss, 
                                       )
        else:
            return self._forward_test(box_regression, centerness, anchors, box_cls, token_logits, dot_product_logits, positive_map, fused_visual_features=fused_visual_features)

    
    
def do_nothing(x, **kwargs):
    return torch.rand(x.shape[0], 256, 4, 11).cuda()

class VLDyHead_RobustSAM(VLDyHead):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        from maskrcnn_benchmark.modeling.backbone.swint_proposed import MaskFeatureBlock, FirstLayerFeatureBlock, LastLayerFeatureBlock, Mlp, TokenBlock
        transformer_dim = 768
        swin_dim = 96
        neck_dim = 256

        if cfg.MODEL.RPN.NO_FUSE == "aggregate" or cfg.MODEL.RPN.NO_FUSE == "aggregate_all":
            self.mode = "aggregate"
            self.robust_token = nn.Parameter( torch.randn(1, transformer_dim) )
            self.process_filters = self.operations
            if cfg.MODEL.RPN.NO_FUSE == "aggregate_all" :
                self.process_filters = self.operations_all
        elif cfg.MODEL.RPN.NO_FUSE == "hidden" or cfg.MODEL.RPN.NO_FUSE == "hidden_all":
            self.mode = "hidden"
            self.robust_token = nn.Parameter( torch.randn(1, neck_dim, transformer_dim) )
            self.process_filters = self.operations
            if cfg.MODEL.RPN.NO_FUSE == "hidden_all" :
                self.process_filters = self.operations_all
            
        # AMFG for mask features
        self.robust_fourier_mask_features = MaskFeatureBlock(transformer_dim=neck_dim)
        # AMFG for image encoder features
        if cfg.MODEL.RPN.NO_FUSE == "aggregate_all" or cfg.MODEL.RPN.NO_FUSE == "hidden_all" :
            self.robust_fourier_first_layer_features = nn.ModuleList([FirstLayerFeatureBlock(vit_dim=swin_dim, transformer_dim=neck_dim, factor=2 ** (i + 1) ) for i in range(5)])
        else:
            self.robust_fourier_first_layer_features = FirstLayerFeatureBlock(vit_dim=swin_dim, transformer_dim=neck_dim)

        self.robust_fourier_last_layer_features = LastLayerFeatureBlock(transformer_dim=neck_dim)
        
        self.AOTG = None 
        if self.AOTG:
            # corresponding new MLP layer for ROT
            self.robust_mlp = Mlp(transformer_dim, transformer_dim//8, neck_dim, ) 
            # AOTG
            self.robust_custom_token_block = TokenBlock(input_dim=neck_dim, mlp_dim=int(transformer_dim // 3), split_dim=3)        

    def operations(self, language_dict_features, indices, x, clear_label=False, shallow_feats=None):
        x = [e[indices] for e in x]
        if not clear_label:
            # torch.Size([30, 96, 104, 336])        
            complementary_features = self.robust_fourier_first_layer_features(shallow_feats[indices], clear=clear_label ) 
            # torch.Size([30, 256, 4, 11])
            final_image_embeddings = self.robust_fourier_last_layer_features(x[-1], clear=clear_label)
            robust_features = complementary_features + final_image_embeddings # fuse image's complementary features and final embeddings 
            
        language_dict_features['embedded'] = language_dict_features['embedded'][indices]
        language_dict_features['aggregate'] = language_dict_features['aggregate'][indices]
        language_dict_features['masks'] = language_dict_features['masks'][indices]
        language_dict_features['hidden'] = language_dict_features['hidden'][indices]

        ##### Image Features after decoder 
        feat_inputs = {"visual": x, "lang": language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        if not clear_label:
            mask_features = self.robust_fourier_mask_features(dyhead_tower['visual'][-1], clear=clear_label)
            # Fusion Feature
            dyhead_tower['visual'][-1] = mask_features + robust_features

        return dyhead_tower

    def operations_all(self, language_dict_features, indices, x, clear_label=False, shallow_feats=None):
        x = [e[indices] for e in x]
        
        language_dict_features['embedded'] = language_dict_features['embedded'][indices]
        language_dict_features['aggregate'] = language_dict_features['aggregate'][indices]
        language_dict_features['masks'] = language_dict_features['masks'][indices]
        language_dict_features['hidden'] = language_dict_features['hidden'][indices]

        ##### Image Features after decoder 
        feat_inputs = {"visual": x, "lang": language_dict_features}
        dyhead_tower = self.dyhead_tower(feat_inputs)

        if not clear_label:            
            for i, e in enumerate(x):  
                # torch.Size([15, 256, 52, 168])
                # torch.Size([15, 256, 26, 84])
                # torch.Size([15, 256, 13, 42])
                # torch.Size([15, 256, 7, 21])
                # torch.Size([15, 256, 4, 11])
                complementary_features = self.robust_fourier_first_layer_features[i](shallow_feats[indices], clear=clear_label ) 
                # torch.Size([30, 96, 104, 336])        
                final_image_embeddings = self.robust_fourier_last_layer_features(e, clear=clear_label)
                robust_features = complementary_features + final_image_embeddings # fuse image's complementary features and final embeddings 

                input_shape = dyhead_tower['visual'][i].shape
                mask_features = self.robust_fourier_mask_features(dyhead_tower['visual'][i], clear=clear_label)
                assert input_shape == mask_features.shape
                dyhead_tower['visual'][i] = mask_features + robust_features

        return dyhead_tower

    def merge_outputs(self, dyhead_tower_clear, dyhead_tower_dirty, clear_labels):
        key = 'visual'
        for i in range(len(dyhead_tower_clear[key])):
            dyhead_tower_clear[key][i] = torch.cat([dyhead_tower_clear[key][i], dyhead_tower_dirty[key][i]], 0 )
                
        key = 'lang'
        for local_key in dyhead_tower_clear[key]:
            if dyhead_tower_clear[key][local_key] is not None:
                dyhead_tower_clear[key][local_key] = torch.cat([dyhead_tower_clear[key][local_key], dyhead_tower_dirty[key][local_key]], 0 )

        # for e in dyhead_tower_clear['visual']:e.requires_grad
        # for e in dyhead_tower_clear['lang']:dyhead_tower_clear['lang'][e].requires_grad
            
        # for e in dyhead_tower_dirty['visual']:e.requires_grad
        # for e in dyhead_tower_dirty['lang']:dyhead_tower_dirty['lang'][e].requires_grad
        return dyhead_tower_clear

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None, clear_labels=False, shallow_feats=None):
        logits = []
        bbox_reg = []
        centerness = []

        # language_dict_features.keys()
        # language_dict_features['aggregate'].shape
        # torch.Size([15, 768])

        # language_dict_features['embedded'].shape
        # torch.Size([15, 256, 768])

        # language_dict_features['masks'].shape
        # torch.Size([15, 256])

        # language_dict_features['hidden'].shape
        # torch.Size([15, 256, 768])

        assert x[-1].requires_grad == False and shallow_feats.requires_grad == False, "Trainable Backbone!!!" 
        assert language_dict_features['hidden'].requires_grad == False and language_dict_features['masks'].requires_grad == False and \
            language_dict_features['embedded'].requires_grad == False and   language_dict_features['aggregate'].requires_grad == False, "Trainable language!!"

        indices_clear = clear_labels == 1
        language_dict_features_dirty = copy.deepcopy(language_dict_features)
        # language_dict_features_dirty[self.mode] = repeat ( self.robust_token, "1 ... -> B ... ", B=len(indices_clear) )
        
        dyhead_tower_clear = self.process_filters(language_dict_features, indices=indices_clear, x=x, clear_label=True, shallow_feats=shallow_feats)
        dyhead_tower_dirty = self.process_filters(language_dict_features_dirty, indices=~indices_clear, x=x, clear_label=False, shallow_feats=shallow_feats,)
        
        # MLM loss
        mlm_logits = None

        # contrastive
        contrastive_logits = None
        proj_tokens = None
        
        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        # soft token
        t_logits = None
        dot_product_logits = []

        if self.AOTG:
            embedding_clear = dyhead_tower_clear["lang"]["hidden"]
            embedding_dirty = dyhead_tower_dirty["lang"]["hidden"]
            # norm
            embedding_clear = F.normalize(embedding_clear, p=2, dim=-1)
            embedding_dirty = F.normalize(embedding_dirty, p=2, dim=-1)
            embedding_dirty = self.robust_custom_token_block(embedding_dirty)

            embedding = torch.cat([embedding_clear, embedding_dirty], 0 )
            dot_product_proj_tokens_clear = self.dot_product_projection_text(embedding_clear / 2.0)
            dot_product_proj_tokens_dirty = self.robust_mlp(embedding_dirty / 2.0)
            dot_product_proj_tokens = torch.cat([dot_product_proj_tokens_clear, dot_product_proj_tokens_dirty], 0 )
            dyhead_tower = self.merge_outputs(dyhead_tower_clear, dyhead_tower_dirty, clear_labels)
        else:
            dyhead_tower = self.merge_outputs(dyhead_tower_clear, dyhead_tower_dirty, clear_labels)
            embedding = dyhead_tower["lang"]["hidden"]
            embedding = F.normalize(embedding, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)

        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0


        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        fused_visual_features = None
        
        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            # torch.Size([15, 4, 52, 168])
            bbox_reg.append(bbox_pred)
            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
                # HERE...... 
                x = dyhead_tower["visual"][l]
                if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                    fused_visual_features.append(x)
                B, C, H, W = x.shape

                # add bias (language)
                dot_product_proj_queries = self.dot_product_projection_image(x)
                dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)

                A = dot_product_proj_queries.shape[1]
                bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)

                dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
                if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
                dot_product_logits.append(dot_product_logit)

        # no matter the feature is from backboone or from fpn, we use shallow_img_embs all the time
        if shallow_img_emb_feats is not None and shallow_text_emb is not None:
            # shallow_img_embs = torch.cat(shallow_img_embs, dim=1)
            proj_tokens = shallow_text_emb
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features


class VLDyHeadModule_RobustSAM(VLDyHeadModule):

    def __init__(self, cfg=None):
        super().__init__(cfg=cfg, )
        
        del self.head
        self.head = VLDyHead_RobustSAM(cfg=cfg)        
        
    def forward(self, images, features, targets=None, language_dict_features=None,
                positive_map=None, captions=None, swint_feature_c4=None, clear_labels=None, shallow_feats=None):

        embedding = language_dict_features['embedded']
        text_masks = language_dict_features["masks"]

        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features = self.head(features,
                                                                        language_dict_features,
                                                                        embedding,
                                                                        swint_feature_c4, 
                                                                        clear_labels,
                                                                        shallow_feats=shallow_feats
                                                                        )
        anchors = self.anchor_generator(images, features)
        
        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors, captions, positive_map, 
                                token_logits, proj_tokens, contrastive_logits, dot_product_logits, text_masks, mlm_logits = mlm_logits,
                                mlm_labels = language_dict_features["mlm_labels"], shallow_img_emb_feats=shallow_img_emb_feats, fused_visual_features=fused_visual_features)
        else:
            return self._forward_test(box_regression, centerness, anchors, box_cls, token_logits, dot_product_logits, 
                    positive_map, fused_visual_features=fused_visual_features)

    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors,
                       captions=None,
                       positive_map=None,
                       token_logits=None,
                       proj_tokens=None,
                       contrastive_logits=None,
                       dot_product_logits=None,
                       text_masks=None,
                       mlm_logits=None,
                       mlm_labels=None,
                       shallow_img_emb_feats=None,
                       fused_visual_features=None
                       ):

        loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors,
            captions,
            positive_map,
            token_logits,
            proj_tokens,
            contrastive_logits,
            dot_product_logits,
            text_masks,
            shallow_img_emb_feats
        )

        losses = {
            # "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

        if mlm_labels is not None and mlm_logits is not None:
            losses["mlm_loss"] = nn.CrossEntropyLoss(ignore_index = -100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)) * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            losses["loss_cls"] = loss_box_cls
        else:
            losses["loss_cls"] = 0.0 * loss_box_cls

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            losses["loss_token"] = loss_token * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            losses["loss_contrastive_align"] = loss_contrastive_align * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            losses["loss_dot_product_token"] = loss_dot_product_token * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS or \
                self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            losses["loss_shallow_contrastive"] = loss_shallow_contrastive * \
                                                 self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT

        if self.cfg.MODEL.RPN_ONLY:
            return None, losses, None
        else:
            # Let's just use one image per batch
            assert (box_regression[0].shape[0]) == 1
            positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)
            boxes = self.box_selector_train(box_regression, centerness, anchors,
                                        box_cls,
                                        token_logits,
                                        dot_product_logits,
                                        positive_map=positive_map_label_to_token
                                        )
            train_boxes = []
            for b, t in zip(boxes, targets):
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                train_boxes.append(cat_boxlist([b, tb]))
            return train_boxes, losses, fused_visual_features



class VLDyHeadModule_Dump(VLDyHeadModule):

    def forward(self, images, features, targets=None, language_dict_features=None,
                positive_map=None, captions=None, swint_feature_c4=None, ):
        

        embedding = language_dict_features['embedded']
        text_masks = language_dict_features["masks"]

        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features = \
            self.head(features, language_dict_features, embedding, swint_feature_c4)

        anchors = self.anchor_generator(images, features)
        boxes, _, _ = self._forward_test(box_regression, centerness, anchors,
                                      box_cls,
                                      token_logits,
                                      dot_product_logits,
                                      positive_map,
                                      fused_visual_features=fused_visual_features
                                      )
        
        return boxes, fused_visual_features
        
        









class VLDyHead_ONLY_FUSE(VLDyHead):
    def __init__(self, cfg):
        cfg.defrost()
        cfg.MODEL.RPN.RETURN_FUSED_FEATURES = True 
        cfg.freeze()
        super().__init__(cfg=cfg)

        self.N_BR = cfg.DATASETS.N_BR # number of points per sample (HD & LQ)
        self.N_LR = cfg.DATASETS.N_LR # no of caption variants per sample
        self.fusion_method = self.fuse_with_vision
        
    def fuse_with_vision(self, x, language_dict_features_new, unique_ids):
        B = unique_ids
        N_x = len(x)
        x_variant = []
        for k in range(N_x) :
            e = repeat(x[k], "B C H W -> (B N) C H W", N=self.N_LR)
            x_variant.append(e)
        
        # print(x_variant[-1].mean(-1).mean(-1).mean(-1))
        feat_inputs_variant = {"visual": x_variant, "lang": language_dict_features_new}
        dyhead_tower_variant = self.dyhead_tower(feat_inputs_variant)        
        return dyhead_tower_variant["visual"][-1]

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None, language_dict_features_new=None ):

        logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, \
        fused_visual_features = super().forward(x, language_dict_features, embedding, swint_feature_c4)


        fused_visual_features = rearrange(fused_visual_features[-1], "B C H W -> B 1 C (H W)")
        fused_visual_features = fused_visual_features.mean(-1)
        # print(fused_visual_features.mean(-1))
        if language_dict_features_new is not None:
            unique_ids = x[-1].shape[0] // self.N_BR
            variant_fused_embedding = self.fusion_method(x, language_dict_features_new, unique_ids=unique_ids)
            variant_fused_embedding = rearrange(variant_fused_embedding, "(B N) C H W -> B N C (H W)", B=unique_ids, N=self.N_LR)
            variant_fused_embedding = variant_fused_embedding.mean(-1)
            # print(variant_fused_embedding.mean(-1))
            
            variant_fused_embedding = torch.cat([variant_fused_embedding, fused_visual_features],1)
            # print(variant_fused_embedding.mean(-1))
            del language_dict_features_new
        else:
            variant_fused_embedding = fused_visual_features 


        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, None, variant_fused_embedding



class VLDyHeadModule_LANG_FUSE(VLDyHeadModule):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        
        del self.head
        self.head = VLDyHead_ONLY_FUSE(cfg=cfg)        
        print("Only Fuse Lang VLDyHeadModule_LANG_FUSE ... ")                  

    def forward(self, images, features, targets=None, language_dict_features=None,
                positive_map=None, captions=None, swint_feature_c4=None,
                language_dict_features_new = None):
        
        
        embedding = language_dict_features['embedded']
        text_masks = language_dict_features["masks"]

        box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features, variant_fused_embedding = self.head( 
            features, language_dict_features, embedding, swint_feature_c4, language_dict_features_new=language_dict_features_new
        )
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       proj_tokens,
                                       contrastive_logits,
                                       dot_product_logits,
                                       text_masks,
                                       mlm_logits = mlm_logits,
                                       mlm_labels = language_dict_features["mlm_labels"],
                                       shallow_img_emb_feats=shallow_img_emb_feats,
                                       fused_visual_features=fused_visual_features,
                                       ), variant_fused_embedding
        else:
            return self._forward_test(box_regression, centerness, anchors, box_cls, token_logits, dot_product_logits, positive_map, fused_visual_features=fused_visual_features), None 

    
    