import json
import msgpack
import numpy as np
from tqdm import tqdm

from pathlib import Path

result_dir = Path("result_records_msgpack")

# for result_file in result_dir.glob("*.json"):
#     with open(result_file, "r") as f:
#         data = json.load(f)
    
IS_REFINED = [
    "FALSE",
    "TRUE",
]

MODELS = [
    "capeformer",
    "graphcape",
    "edgecape",
]

SPLITS = [
    "1",
    "4",
]

N_SAMPLES = [
    5,
    10,
    20,
    30,
    50,
    100,
]

N_QUERIES = [
    1
]

N_EPISODES = [
    300,
]


USE_PROTOTYPE = [
    # "FALSE",
    "TRUE",
]

USE_ORIGINAL = [
    "FALSE",
    "TRUE",
]

MASK_PERCENT = [
    0,
    25,
    50,
    65,
    75,
    90,
    95,
]

MASK_AFTER = [
    "FALSE",
    "TRUE",
]

def pck(acc, gt, pred, bbox, thresholds, mask):
    max_side = max(bbox[2], bbox[3])
    error = np.linalg.norm((gt - pred) / max_side, axis=1)[mask]
    # assert np.sum(error > 1000) == 0, "Some keypoints have error > 1000"
    if np.sum(error > 1000) > 0:
        print("Warning: Some keypoints have error > 1000")
    for threshold in thresholds:
        acc[threshold].append((error < threshold).mean())

threshold_list = [.05, .1, .15, .2, .25]
def report_result(data, n_eps, ori):
    assert len(data) == n_eps * 1000, f"Expected {n_eps * 1000} results, but got {len(data)}"

    start_kpts_idx = 68
    
    acc_results = []

    acc = {}
    for threshold in threshold_list:
        acc[threshold] = []

    if ori:
        for result in tqdm(data):
            bbox = np.array(result['bbox'])
            gt = np.array(result['gt'])[start_kpts_idx:]
            pred = np.array(result['pred'])[start_kpts_idx:]
            if 'mask' in result:
                mask = np.array(result['mask'])[start_kpts_idx:]
            else:
                mask = np.ones_like(gt[:,0], dtype=bool)
            pck(acc, gt, pred, bbox, threshold_list, mask)
        for threshold in threshold_list:
            acc_results.append(np.mean(acc[threshold]))
            acc[threshold] = []
        for result in tqdm(data):
            bbox = np.array(result['bbox'])
            gt = np.array(result['gt'])[:start_kpts_idx]
            pred = np.array(result['pred'])[:start_kpts_idx]
            if 'mask' in result:
                mask = np.array(result['mask'])[:start_kpts_idx]
            else:
                mask = np.ones_like(gt[:,0], dtype=bool)
            pck(acc, gt, pred, bbox, threshold_list, mask)
        for threshold in threshold_list:
            acc_results.append(np.mean(acc[threshold]))
            acc[threshold] = []
    else:
        for result in tqdm(data):
            bbox = np.array(result['bbox'])
            gt = np.array(result['gt'])
            pred = np.array(result['pred'])
            if 'mask' in result:
                mask = np.array(result['mask'])
            else:
                mask = np.ones_like(gt[:,0], dtype=bool)
            pck(acc, gt, pred, bbox, threshold_list, mask)
        for threshold in threshold_list:
            acc_results.append(np.mean(acc[threshold]))
            acc[threshold] = []
        for threshold in threshold_list:
            acc_results.append('')

    return acc_results

output_path = Path("summary_results.txt")
if output_path.exists():
    with open(output_path, "r") as f:
        lines = f.readlines()
        current_data = [line.strip().split("\t") for line in lines]
        current_data = {
            line[-1]: line for line in current_data
        }
else:
    current_data = None
    

output_file = open(output_path, "w")

for model in MODELS:
    for split in SPLITS:
        for n_sample in N_SAMPLES:
            for n_query in N_QUERIES:
                for n_episode in N_EPISODES:
                    for is_refined in IS_REFINED:
                        for use_prototype in USE_PROTOTYPE:
                            for use_original in USE_ORIGINAL:
                                for mask_p in MASK_PERCENT:
                                    for mask_after in MASK_AFTER:
                                        if is_refined == "TRUE" and use_original == "FALSE":
                                            continue
                                        if use_original == "FALSE" and mask_p > 0:
                                            continue
                                        if is_refined == "FALSE" and mask_after == "TRUE":
                                            continue
                                        result_file_name = f"result_voronoi{'_ref' if is_refined == 'TRUE' else ''}_{model}_s{split}_{n_sample}_{n_query}_{n_episode}k{'_prototype' if use_prototype == 'TRUE' else ''}{'_ori' if use_original == 'TRUE' else ''}{'_m'+str(mask_p)+'p' if mask_p > 0 else ''}{'_abl' if mask_after == 'TRUE' else ''}"
                                        result_file = result_dir / f"{result_file_name}.msgpack"
                                        if current_data is not None:
                                            if result_file_name in current_data and current_data[result_file_name][8] == "Done":
                                                print(f"Skipping {current_data[result_file_name]}")
                                                output_file.write("\t".join(current_data[result_file_name]) + "\n")
                                                output_file.flush()
                                                continue
                                        if result_file.exists():
                                            print(f"Processing {result_file}")
                                            with open(result_file, "rb") as f:
                                                data = msgpack.unpack(f, raw=False)
                                            accs = report_result(data, n_episode, use_original == "TRUE")
                                            options = [model, split, n_sample, is_refined, use_prototype, use_original, mask_p, mask_after, "Done"]
                                            output_file.write("\t".join(map(str, options + accs + [result_file_name])) + "\n")
                                            output_file.flush()
                                        else:
                                            print(f"File {result_file} does not exist")
                                            options = [model, split, n_sample, is_refined, use_prototype, use_original, mask_p, mask_after, "Not Started"]
                                            output_file.write("\t".join(map(str, options + [''] * (2*len(threshold_list)) + [result_file_name])) + "\n")
                                            output_file.flush()

output_file.close()