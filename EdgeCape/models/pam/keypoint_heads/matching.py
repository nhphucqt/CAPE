from copy import deepcopy
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models import HEADS
from mmpose.core.post_processing import transform_preds

@HEADS.register_module()
class MatchingHead(nn.Module):
    '''
    In two stage regression A3, the proposal generator are moved into transformer.
    All valid proposals will be added with an positional embedding to better regress the location
    '''

    def __init__(self, 
                 img_size,
                 train_cfg=None,
                 test_cfg=None,):
        super().__init__()

        self.img_size = img_size
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatMap')

        self.resize = partial(F.interpolate,
            size=(img_size, img_size), mode='bilinear', align_corners=False
        )

    def forward(self, x, feature_s, target_s, mask_s, skeleton):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """

        x = self.resize(x).flatten(2) # [bs, c, h*w]

        bs, c, hw = x.shape
        h, w = self.img_size, self.img_size
        assert hw == h * w, f"input feature has wrong size {hw}, expected {h*w}"

        feature_s = [self.resize(feature) for feature in feature_s]
        target_s = [self.resize(target) for target in target_s]
        # process keypoint token feature
        query_embed_list = []
        for i, (feature, target) in enumerate(zip(feature_s, target_s)):
            # resize the support feature back to the heatmap sizes.
            target = target / (target.sum(dim=-1).sum(dim=-1)[:, :, None, None] + 1e-8)
            support_keypoints = target.flatten(2) @ feature.flatten(2).permute(0, 2, 1)
            query_embed_list.append(support_keypoints)

        # print(f"{len(query_embed_list) = }")
        # print(f"{query_embed_list[0].shape = }")

        support_keypoints = torch.mean(torch.stack(query_embed_list, dim=0), 0)
        support_keypoints = support_keypoints * mask_s

        support_keypoints = support_keypoints.flatten(2) # [bs, num_kpt, c]

        # Create adjacency matrix for keypoints based on skeleton
        num_keypoints = support_keypoints.shape[1]
        adj = np.zeros((num_keypoints, num_keypoints), dtype=np.float32)
        # print(f"{skeleton = }")
        for sk in skeleton[0]:
            i, j = sk
            adj[i, j] = 1
            adj[j, i] = 1  # assuming undirected edges
        # print(f"{adj = }")
        # print(f"{np.identity(num_keypoints)[None].shape = }")
        adj = np.concatenate((np.identity(num_keypoints)[None], adj[None]), axis=0)
        adj = torch.from_numpy(adj).to(feature_s[0].device)
        adj = adj.unsqueeze(0)

        # print(f"{support_keypoints.shape = }")
        # print(f"{x.shape = }")

        # --- Matching step ---
        # similarity: [bs, num_kpt, h*w]
        similarity = torch.bmm(support_keypoints, x)

        # print(f"{similarity.shape = }")

        # # reshape to spatial map: [bs, num_kpt, h, w]

        # find max location (proposal keypoint)
        max_idx = similarity.argmax(dim=-1)  # [bs, num_kpt]

        similarity_map = similarity.reshape(bs, -1, h, w)

        # print(f"{max_idx.shape = }")
        # print(f"{similarity_map.shape = }")

        y_coords = max_idx // w
        x_coords = max_idx % w

        # print(f"{x_coords.shape = }")
        # print(f"{y_coords.shape = }")

        # normalize to [0, 1]
        x_coords = x_coords.float() / w
        y_coords = y_coords.float() / h

        # stack into proposal keypoints [bs, num_kpt, 2]
        proposal_keypoints = torch.stack([x_coords, y_coords], dim=-1)

        # print(f"{proposal_keypoints.shape = }")

        return proposal_keypoints[None], proposal_keypoints, similarity_map, adj

        # return torch.stack(output_kpts, dim=0), initial_proposals, similarity_map, adj

    def decode(self, img_metas, output, img_size, **kwargs):
        """Decode the predicted keypoints from prediction.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)
        W, H = img_size
        output = output * np.array([W, H])[None, None, :]  # [bs, query, 2], coordinates with recovered shapes.

        if 'bbox_id' or 'query_bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['query_center']
            s[i, :] = img_metas[i]['query_scale']
            image_paths.append(img_metas[i]['query_image_file'])

            if 'query_bbox_score' in img_metas[i]:
                score[i] = np.array(
                    img_metas[i]['query_bbox_score']).reshape(-1)
            if 'bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['bbox_id'])
            elif 'query_bbox_id' in img_metas[i]:
                bbox_ids.append(img_metas[i]['query_bbox_id'])

        preds = np.zeros(output.shape)
        for idx in range(output.shape[0]):
            preds[idx] = transform_preds(
                output[idx],
                c[idx],
                s[idx], [W, H],
                use_udp=self.test_cfg.get('use_udp', False))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = 1.0  # NOTE: Currently, assume all predicted points are of 100% confidence.
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result
