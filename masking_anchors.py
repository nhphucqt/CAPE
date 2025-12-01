# %%
from xtcocotools.coco import COCO
from matplotlib import pyplot as plt
import numpy as np
import json
from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay
import msgpack
import torch
from tqdm import tqdm
from pathlib import Path

# %%
from EdgeCape.models.keypoint_heads.refine_head import RefineHead

refine_head = RefineHead()

# %%
MODEL_NAMES = [
    # 'capeformer',
    # 'graphcape',
    'edgecape',
]

MASK_PERCENTS = [
    "",
    # "_m25p",
    "_m50p",
    "_m75p",
    # "_m65p",
    "_m90p",
]

# %%
model_name = "graphcape"

for model_name in MODEL_NAMES:
    for mask_percent in MASK_PERCENTS:
        file_name = f"result_voronoi_{model_name}_s1_5_1_300k_prototype_ori{mask_percent}"
        out_name = f"result_voronoi_ref_{model_name}_s1_5_1_300k_prototype_ori{mask_percent}"
        if not Path(f"result_records_msgpack/{out_name}.msgpack").exists():
            print(f"{file_name} does not exist, skip")
            continue
        print(f"Processing {file_name} -> {out_name}")
        with open(f"result_records_msgpack/{file_name}.msgpack", "rb") as f:
            result_keypoints = msgpack.load(f)

        for result in tqdm(result_keypoints):
            # target_s
            # preds
            # coarse
            # mask_s
            target_s = np.array(result["target_s"])[None]
            preds = np.array(result["pred"])[None, None]
            coarse = np.array([1.0] * 68 + [0.0] * 5)[None]
            if "mask" not in result:
                mask_s = np.array([1.0] * 73)[None].astype(np.float32)
            else:
                mask_s = np.array(result["mask"])[None].astype(np.float32)

            # print(mask_s)

            target_s = torch.tensor(target_s).float()
            preds = torch.tensor(preds).float()
            coarse = torch.tensor(coarse).float()
            mask_s = torch.tensor(mask_s).float()

            # print(target_s.shape, preds.shape, coarse.shape, mask_s.shape)
            # print(mask_s)

            # print(preds[0].squeeze(0).numpy()[68:])
            out = refine_head(target_s, preds, coarse, mask_s)
            # print(out[0].squeeze(0).numpy()[68:])
            # print(out[-1].squeeze(0).numpy().shape)
            result["pred"] = out[-1].squeeze(0).numpy().tolist()

        with open(f"{out_name}.msgpack", "wb") as f:
            msgpack.dump(result_keypoints, f)

# # %%
# start_kpts_idx = 68
# threshold_list = [.05, .1, .15, .2, .25]
# acc = {}
# for threshold in threshold_list:
#     acc[threshold] = []
# print("Total images:", len(result_keypoints))
# for id, result in enumerate(result_keypoints):
#     gt = np.array(result['gt'])[start_kpts_idx:]
#     pred = np.array(result['pred'])[start_kpts_idx:]
#     if 'mask' in result:
#         mask = np.array(result['mask'])[start_kpts_idx:]
#     else:
#         mask = np.ones_like(gt[:, 0]).astype(bool)
#     # gt = np.array(result['gt'])[:start_kpts_idx]
#     # pred = np.array(result['pred'])[:start_kpts_idx]
#     bbox = np.array(result['bbox'])
#     max_side = max(bbox[2], bbox[3])
#     # error = np.linalg.norm(gt - pred, axis=1) / max_side
#     error = np.linalg.norm((gt - pred) / max_side, axis=1)[mask]
#     if np.sum(error > 1000) > 0:
#         print(f"Warning: image {id} has {np.sum(error > 1000)} keypoints with error > 1000")
#         continue
#     for threshold in threshold_list:
#         acc[threshold].append((error < threshold).mean())
# for threshold in threshold_list:
#     print(f"PCK@{threshold}:", np.mean(acc[threshold]))

# # %%
# with open(f"{out_name}.msgpack", "wb") as f:
#     msgpack.dump(result_keypoints, f)

# # %%
# # Total images: 300000
# # PCK@0.05: 0.9294966666666669
# # PCK@0.1: 0.9965980000000001
# # PCK@0.15: 0.9995073333333334
# # PCK@0.2: 0.9998466666666669
# # PCK@0.25: 0.9999300000000002

# # %%



