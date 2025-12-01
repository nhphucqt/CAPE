from mmpose.datasets import DATASETS
import random
import numpy as np
import os
from collections import OrderedDict
from xtcocotools.coco import COCO
from .test_base_dataset import TestBaseDataset
from mmcv.parallel import DataContainer as DC
import msgpack

import json
import copy
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay
from datetime import datetime
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme, 
                                                  keypoint_pck_accuracy)

@DATASETS.register_module()
class TestPoseVoronoiDataset(TestBaseDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 valid_class_ids,
                 max_kpt_num=None,
                 use_prototype=True,
                 keep_orig=False,
                 ori_mask_percent=0.0,
                 num_shots=1,
                 num_sample_kpts=3,
                 num_queries=100,
                 num_episodes=1,
                 pck_threshold_list=[0.05, 0.1, 0.15, 0.20, 0.25],
                 test_mode=True):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode, PCK_threshold_list=pck_threshold_list)

        self.ann_info['flip_pairs'] = []

        self.ann_info['upper_body_ids'] = []
        self.ann_info['lower_body_ids'] = []

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array([1.,],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.coco = COCO(ann_file)

        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.img_ids = self.coco.getImgIds()
        self.classes = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]

        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, self.coco.getCatIds()))
        self._ind_to_class = dict(zip(self.coco.getCatIds(), self.classes))

        if valid_class_ids is not None: # None by default
            self.valid_class_ids = valid_class_ids
        else:
            self.valid_class_ids = self.coco.getCatIds()
        self.valid_classes = [self._ind_to_class[ind] for ind in self.valid_class_ids]

        self.cats = self.coco.cats
        self.max_kpt_num = max_kpt_num

        # Also update self.cat2obj
        self.db = self._get_db()

        self.num_shots = num_shots

        assert keep_orig or ori_mask_percent == 0.0, "Cannot mask original keypoints if they are not kept"
        self.use_prototype = use_prototype
        self.keep_orig = keep_orig
        self.num_sample_kpts = num_sample_kpts
        self.ori_mask_percent = ori_mask_percent

        if not test_mode:
            # Update every training epoch
            self.random_paired_samples()
        else:
            self.num_queries = num_queries
            self.num_episodes = num_episodes
            self.make_paired_samples()
            self.prepare_modified_annotations()

    def random_paired_samples(self):
        num_datas = [len(self.cat2obj[self._class_to_ind[cls]]) for cls in self.valid_classes]

        # balance the dataset
        max_num_data = max(num_datas)

        all_samples = []
        for cls in self.valid_class_ids:
            for i in range(max_num_data):
                shot = random.sample(self.cat2obj[cls], self.num_shots + 1)
                all_samples.append(shot)

        self.paired_samples = np.array(all_samples)
        np.random.shuffle(self.paired_samples)

    def make_paired_samples(self):
        random.seed(1)
        np.random.seed(0)

        all_samples = []
        for cls in self.valid_class_ids:
            for _ in range(self.num_episodes):
                shots = random.sample(self.cat2obj[cls], self.num_shots + self.num_queries)
                sample_ids = shots[:self.num_shots]
                query_ids = shots[self.num_shots:]
                for query_id in query_ids:
                    all_samples.append(sample_ids + [query_id])
                    # all_samples.append(sample_ids + [sample_ids[0]])  # For debugging, use the first support as the query

        self.paired_samples = np.array(all_samples)

    def _select_kpt(self, obj, kpt_id):
        obj['joints_3d'] = obj['joints_3d'][kpt_id:kpt_id+1]
        obj['joints_3d_visible'] = obj['joints_3d_visible'][kpt_id:kpt_id+1]
        obj['kpt_id'] = kpt_id

        return obj

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_db(self):
        """Ground truth bbox and keypoints."""
        self.obj_id = 0

        self.cat2obj = {}
        for i in self.coco.getCatIds():
            self.cat2obj.update({i: []})

        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue

            category_id = obj['category_id']
            # the number of keypoint for this specific category
            cat_kpt_num = int(len(obj['keypoints']) / 3)
            if self.max_kpt_num is None:
                kpt_num = cat_kpt_num
            else:
                kpt_num = self.max_kpt_num

            joints_3d = np.zeros((kpt_num, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((kpt_num, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:cat_kpt_num, :2] = keypoints[:, :2]
            joints_3d_visible[:cat_kpt_num, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])

            self.cat2obj[category_id].append(self.obj_id)

            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'bbox': obj['clean_bbox'][:4],
                'bbox_score': 1,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'category_id': category_id,
                'cat_kpt_num': cat_kpt_num,
                'bbox_id': self.obj_id,
                'skeleton': self.coco.cats[obj['category_id']]['skeleton'],
            })
            bbox_id = bbox_id + 1
            self.obj_id += 1

        return rec

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info['image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        #
        # if (not self.test_mode) and np.random.rand() < 0.3:
        #     center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * 1.25

        return center, scale

    def _report_metric(self,
                       res_file,
                       metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.paired_samples)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, pair, m_pair in zip(preds, self.paired_samples, self.modified_pairs):
            item = self.db[pair[-1]]
            m_item = m_pair[1]
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(m_item['joints_3d'])[:, :-1])

            mask_query = ((np.array(m_item['joints_3d_visible'])[:, 0]) > 0)
            mask_sample = ((np.array(m_pair[0][0]['joints_3d_visible'])[:, 0]) > 0)
            for src in m_pair[0]:
                mask_sample = np.bitwise_and(mask_sample, ((np.array(src['joints_3d_visible'])[:, 0]) > 0))
            masks.append(np.bitwise_and(mask_query, mask_sample))

            if 'PCK' in metrics or 'NME' in metrics or 'AUC' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in metrics:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))
        
        if 'PCK' in metrics:
            pck_results = dict()
            for pck_thr in self.PCK_threshold_list:
                pck_results[pck_thr] = []

            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                for pck_thr in self.PCK_threshold_list:
                    _, pck, cnt = keypoint_pck_accuracy(np.expand_dims(output, 0), np.expand_dims(gt,0), np.expand_dims(mask,0), pck_thr, np.expand_dims(thr_bbox,0))
                    # print(f"{pck_thr}: {pck}, Count: {cnt}")
                    # print(f"Valid Keypoints: {cnt}")
                    pck_results[pck_thr].append(pck)

            mPCK = 0
            for pck_thr in self.PCK_threshold_list:
                info_str.append(['PCK@' + str(pck_thr), np.mean(pck_results[pck_thr])])
                mPCK += np.mean(pck_results[pck_thr])
            info_str.append(['mPCK', mPCK / len(self.PCK_threshold_list)])
        
        if 'NME' in metrics:
            nme_results = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                nme = keypoint_nme(np.expand_dims(output, 0), np.expand_dims(gt,0), np.expand_dims(mask,0), np.expand_dims(thr_bbox,0))
                nme_results.append(nme)
            info_str.append(['NME', np.mean(nme_results)])

        if 'AUC' in metrics:
            auc_results = []
            for (output, gt, mask, thr_bbox) in zip(outputs, gts, masks, threshold_bbox):
                auc = keypoint_auc(np.expand_dims(output, 0), np.expand_dims(gt,0), np.expand_dims(mask,0), thr_bbox[0])
                auc_results.append(auc)
            info_str.append(['AUC', np.mean(auc_results)])
            
        if 'EPE' in metrics:
            epe_results = []
            for (output, gt, mask) in zip(outputs, gts, masks):
                epe = keypoint_epe(np.expand_dims(output, 0), np.expand_dims(gt,0), np.expand_dims(mask,0))
                epe_results.append(epe)
            info_str.append(['EPE', np.mean(epe_results)])
        return info_str

    def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
        """Evaluate interhand2d keypoint results. The pose prediction results
        will be saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_paths (list[str]): For example, ['C', 'a', 'p', 't',
                    'u', 'r', 'e', '1', '2', '/', '0', '3', '9', '0', '_',
                    'd', 'h', '_', 't', 'o', 'u', 'c', 'h', 'R', 'O', 'M',
                    '/', 'c', 'a', 'm', '4', '1', '0', '2', '0', '9', '/',
                    'i', 'm', 'a', 'g', 'e', '6', '2', '4', '3', '4', '.',
                    'j', 'p', 'g']
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE', 'NME']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        assert len(self.paired_samples) == len(self.modified_pairs), "Number of modified pairs does not match number of paired samples"

        kpts = []
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        results = []
        for pred, pair, m_pair in zip(kpts, self.paired_samples, self.modified_pairs):
            item = self.db[pair[-1]]
            m_item = m_pair[1]

            bbox = np.array(item['bbox'])
            bbox_thr = np.max(bbox[2:])

            mask_sample = ((np.array(m_pair[0][0]['joints_3d_visible'])[:, 0]) > 0)
            for src in m_pair[0]:
                mask_sample = np.bitwise_and(mask_sample, ((np.array(src['joints_3d_visible'])[:, 0]) > 0))
            
            kpt_num = m_item['cat_kpt_num']

            results.append({
                'pred': np.array(pred['keypoints'])[:kpt_num, :-1].tolist(),
                'gt': np.array(m_item['joints_3d'])[:kpt_num, :-1].tolist(),
                'mask': mask_sample[:kpt_num].tolist(),
                'bbox': bbox.tolist(),
                'img_s': [self.db[id]['image_file'] for id in pair[:-1]],
                'img_q': item['image_file'],
                'target_s': [src['joints_3d'][:kpt_num, :-1].tolist() for src in m_pair[0]],
                'kpt_num': kpt_num,
                'weights': m_item['weights'],
                'skeleton': item['skeleton'],
            })
        # self._write_keypoint_results(results, "result_voronoi.json")
        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # voronoi_file = f"result_voronoi_{dt_str}.json"
        voronoi_file = f"result_voronoi_{dt_str}.msgpack"
        with open(voronoi_file, 'wb') as f:
            msgpack.pack(results, f)
        # with open(voronoi_file.replace('.msgpack', '.json'), 'w') as f:
        #     json.dump(results, f)
        print(f"\nVoronoi results saved to {voronoi_file}")
        
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    def find_centroid_weight(self, keypoints, centroids):
        # keypoints: (N, 2) array of 2D points
        # centroids: (M, 2) array of centroid points

        keypoints = np.asarray(keypoints)
        centroids = np.asarray(centroids)

        tri = Delaunay(keypoints)
        results = []

        for centroid in centroids:
            simplex = tri.find_simplex(centroid)
            if simplex == -1:
                raise ValueError("Centroid is not inside any triangle formed by keypoints.")

            triangle_indices = tri.simplices[simplex]
            triangle_points = keypoints[triangle_indices]

            # Compute barycentric coordinates
            T = np.vstack((triangle_points.T, np.ones(3)))
            C = np.append(centroid, 1)
            weights = np.linalg.solve(T, C)

            results.append((triangle_indices.tolist(), weights.tolist()))

        return results

    def random_sampling(self, keypoints, k, use_prototype=False):
        if use_prototype:
            keypoints = np.array([[0, 45], [2, 75], [8, 104], [16, 133], [26, 158], [41, 180], [56, 200], [81, 214], [114, 222], [147, 214], [172, 200], [187, 180], [202, 158], [212, 133], [220, 104], [226, 75], [228, 45], [15, 10], [33, 4], [51, 0], [71, 2], [91, 8], [137, 8], [157, 2], [177, 0], [195, 4], [213, 10], [114, 40], [114, 56], [114, 72], [114, 88], [91, 100], [103, 103], [114, 104], [125, 103], [137, 100], [36, 42], [51, 32], [71, 32], [86, 42], [71, 52], [51, 52], [142, 42], [157, 32], [177, 32], [192, 42], [177, 52], [157, 52], [69, 154], [79, 140], [97, 135], [114, 136], [131, 135], [149, 140], [159, 154], [149, 168], [131, 172], [114, 175], [97, 172], [79, 168], [79, 154], [97, 149], [114, 147], [131, 149], [149, 154], [131, 161], [114, 164], [97, 161]])
            bound_kp_ids = [i for i in range(17)] + [26, 25, 24, 19, 18, 17]
        else:
            bound_kp_ids = [i for i in range(17)] + [i for i in range(26, 16, -1)]

        bound_kps = keypoints[bound_kp_ids]

        boundary_polygon = Polygon(bound_kps)

        # Get bounding box of boundary keypoints
        min_x, min_y = bound_kps.min(axis=0)
        max_x, max_y = bound_kps.max(axis=0)

        samples = []
        while len(samples) < k:
            # Sample random point in bounding box
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if boundary_polygon.contains(Point(x, y)):
                samples.append([x, y])
        samples = np.array(samples)
        weights = self.find_centroid_weight(keypoints, samples)
        return samples, weights

    def modify_annotation(self, db, weights, is_query, keep_orig=False):
        kpt_num = db['cat_kpt_num']
        keypoints = db['joints_3d'][:kpt_num, :2]
        visible = db['joints_3d_visible'][:kpt_num, :2]
        skeleton = db['skeleton']

        samples = []
        sample_visible = []
        for id, weight in enumerate(weights):
            idxs, bary_weights = weight
            # check if all keypoints are visible
            if not all(visible[idxs, 0] > 0):
                sample_visible.append([0., 0.])
                sample = np.array([0., 0.])
                print(f"Sample {id} has invisible keypoints, setting visibility to 0")
            else:
                sample_visible.append([1., 1.])
                sample = np.dot(bary_weights, keypoints[idxs])
            samples.append(sample)
        samples = np.array(samples)
        sample_visible = np.array(sample_visible)
        k = len(weights)
        target_coarse = np.array([[0.] for _ in range(k)])

        if keep_orig:
            if not is_query and self.ori_mask_percent > 0.0:
                # print("Randomly masking original keypoints")
                # create a random mask (exactly ori_mask_percent percent points are removed) to mask visible array with shape (kpt_num,3)
                # ensure that there are at least 3 unique points visible
                min_n_points = 3
                n_mask = int(np.round(self.ori_mask_percent * kpt_num))
                assert n_mask <= kpt_num - min_n_points, f"Cannot mask {n_mask} points, only {kpt_num} points available"
                n_tries = 0
                while True:
                    visible_1d = visible[:, 0].copy()
                    mask_ids = random.sample(range(kpt_num), n_mask)
                    visible_1d[mask_ids] = 0
                    if len(np.unique(keypoints[visible_1d > 0], axis=0)) >= min_n_points:
                        break
                    n_tries += 1
                    if n_tries > 1000:
                        raise ValueError("Cannot find a valid mask after 1000 tries, please reduce the mask percentage or the minimum number of points")
                visible = visible.copy()
                visible[mask_ids, :] = 0
                
            samples = np.vstack((keypoints, samples))
            sample_visible = np.vstack((visible, sample_visible))
            target_coarse = np.vstack(([[1.] for _ in range(kpt_num)], target_coarse))
            k += kpt_num
        else:
            skeleton = []

        # print(f"{samples.shape = }, {sample_visible.shape = }, {k = }")

        # Replace joints_3d and visible with samples
        new_keypoints = np.zeros_like(db['joints_3d'])
        new_visible = np.zeros_like(db['joints_3d_visible'])
        new_coarse = np.zeros((self.max_kpt_num, 1), dtype=db['joints_3d'].dtype)
        new_keypoints[:k, :2] = samples
        new_visible[:k, :2] = sample_visible
        new_coarse[:k, :] = target_coarse
        
        db['joints_3d'] = new_keypoints
        db['joints_3d_visible'] = new_visible
        db['cat_kpt_num'] = k
        db['skeleton'] = skeleton
        db['target_coarse'] = new_coarse

        return db

    def prepare_modified_annotations(self):
        """Prepare modified annotations for evaluation."""
        self.modified_pairs = []
        for pair_ids in self.paired_samples:
            assert len(pair_ids) == self.num_shots + 1
            sample_id_list = pair_ids[:self.num_shots]
            query_id = pair_ids[-1]
            sample_obj_list = []
            for sample_id in sample_id_list:
                sample_obj = copy.deepcopy(self.db[sample_id])
                sample_obj_list.append(sample_obj)
            query_obj = copy.deepcopy(self.db[query_id])

            _, weights = self.random_sampling(sample_obj_list[0]['joints_3d'][:sample_obj_list[0]['cat_kpt_num'], :2], k=self.num_sample_kpts, use_prototype=self.use_prototype)
            for i in range(self.num_shots):
                sample_obj_list[i] = self.modify_annotation(sample_obj_list[i], weights, is_query=False, keep_orig=self.keep_orig)
            query_obj = self.modify_annotation(query_obj, weights, is_query=True, keep_orig=self.keep_orig)
            assert weights is not None, "Weights should be computed from the first support image"
            self.modified_pairs.append((
                [copy.deepcopy({
                    'joints_3d': sample_obj['joints_3d'],
                    'joints_3d_visible': sample_obj['joints_3d_visible'],
                    'cat_kpt_num': sample_obj['cat_kpt_num'],
                    'skeleton': sample_obj['skeleton'],
                    'target_coarse': sample_obj['target_coarse'],
                    'weights': weights,
                }) for sample_obj in sample_obj_list],
                copy.deepcopy({
                    'joints_3d': query_obj['joints_3d'],
                    'joints_3d_visible': query_obj['joints_3d_visible'],
                    'cat_kpt_num': query_obj['cat_kpt_num'],
                    'skeleton': query_obj['skeleton'],
                    'target_coarse': query_obj['target_coarse'],
                    'weights': weights,
                })
            ))
            # print(f"{len(self.modified_pairs) = }")

    def apply_modified_annotations(self, obj, m_obj):
        obj['joints_3d'] = copy.deepcopy(m_obj['joints_3d'])
        obj['joints_3d_visible'] = copy.deepcopy(m_obj['joints_3d_visible'])
        obj['cat_kpt_num'] = copy.deepcopy(m_obj['cat_kpt_num'])
        obj['skeleton'] = copy.deepcopy(m_obj['skeleton'])
        obj['target_coarse'] = copy.deepcopy(m_obj['target_coarse'])
        return obj

    def __getitem__(self, idx):
        """Get the sample given index."""

        pair_ids = self.paired_samples[idx] # [supported id * shots, query id]
        assert len(pair_ids) == self.num_shots + 1
        sample_id_list = pair_ids[:self.num_shots]
        query_id = pair_ids[-1]

        sample_obj_list = []
        for sample_id in sample_id_list:
            sample_obj = copy.deepcopy(self.db[sample_id])
            sample_obj['ann_info'] = copy.deepcopy(self.ann_info)
            sample_obj_list.append(sample_obj)

        query_obj = copy.deepcopy(self.db[query_id])
        query_obj['ann_info'] = copy.deepcopy(self.ann_info)

        # Modify the annotation to get k random keypoints within the face region
        for i in range(self.num_shots):
            sample_obj_list[i] = self.apply_modified_annotations(sample_obj_list[i], self.modified_pairs[idx][0][i])
        query_obj = self.apply_modified_annotations(query_obj, self.modified_pairs[idx][1])
        # print(f"{query_obj = }")

        Xs_list = []
        for sample_obj in sample_obj_list:
            Xs = self.pipeline(sample_obj) # dict with ['img', 'target', 'target_weight', 'img_metas'], 
            Xs_list.append(Xs)             # Xs['target'] is of shape [100, map_h, map_w]
        Xq = self.pipeline(query_obj)

        # print(f"{Xq = }")

        Xall = self._merge_obj(Xs_list, Xq, idx)
        Xall['skeleton'] = query_obj['skeleton']

        return Xall

    def _merge_obj(self, Xs_list, Xq, idx):
        """ merge Xs_list and Xq.

        :param Xs_list: N-shot samples X
        :param Xq: query X
        :param idx: id of paired_samples
        :return: Xall
        """
        Xall = dict()
        Xall['img_s'] = [Xs['img'] for Xs in Xs_list]
        Xall['target_s'] = [Xs['target'] for Xs in Xs_list]
        Xall['target_weight_s'] = [Xs['target_weight'] for Xs in Xs_list]
        Xall['target_coarse_s'] = [Xs['target_coarse'] for Xs in Xs_list]
        xs_img_metas = [Xs['img_metas'].data for Xs in Xs_list]

        Xall['img_q'] = Xq['img']
        Xall['target_q'] = Xq['target']
        Xall['target_weight_q'] = Xq['target_weight']
        Xall['target_coarse_q'] = Xq['target_coarse']
        xq_img_metas = Xq['img_metas'].data

        img_metas = dict()
        for key in xq_img_metas.keys():
            img_metas['sample_' + key] = [xs_img_meta[key] for xs_img_meta in xs_img_metas]
            img_metas['query_' + key] = xq_img_metas[key]
        img_metas['bbox_id'] = idx

        Xall['img_metas'] = DC(img_metas, cpu_only=True)

        return Xall