#%%

import argparse
from typing import Dict, List
from xtcocotools.coco import COCO
from pathlib import Path
import numpy as np
import random
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
import json

from settings import (
    DATA_ANNOTATIONS_DIR,
    DATA_IMAGES_DIR,
)

random.seed(0)
np.random.seed(0)

num_random_points = 50

# polygons = [
#     [0, 1, 2, 3, 4, 30, 29, 28, 27],
#     [27, 28, 29, 30, 12, 13, 14, 15, 16],
#     [5, 6, 7, 8, 9, 10, 11],
# ]

polygons = [
    [48, 39, 36],
    [54, 45, 42],
    [31, 30, 27, 39],
    [35, 42, 27, 30],
    [4, 48, 36, 0],
    [12, 16, 45, 54],
]

#%%

def load_annotations(annotations_dir: Path) -> Dict:
    anno = {}

    for annotation_file in annotations_dir.glob("*.json"):
        coco = COCO(annotation_file)
        for img_id in coco.getImgIds():
            img_anno = coco.loadImgs(img_id)[0]
            file_name = img_anno['file_name']
            if file_name not in anno:
                anno[file_name] = []

            anno[file_name].append({
                'img_id': img_id,
                'ann_id': img_anno['id'],
                'annotation_filename': str(annotation_file),
                'annotation_name': str(annotation_file.name).split('.')[0].split('_', 1)[-1],
                'width': img_anno['width'],
                'height': img_anno['height'],
                'objs': [],
            })
            ann_ids = coco.getAnnIds(imgIds=img_id)
            objs = coco.loadAnns(ann_ids)

            for obj in objs:
                x, y, width, height = obj['bbox']
                anno[file_name][-1]['objs'].append({
                    'id': obj['id'],
                    'bbox': [x, y, width, height],
                    'num_keypoints': obj['num_keypoints'],
                    'iscrowd': obj['iscrowd'],
                    'area': obj['area'],
                    'category_id': obj['category_id'],
                    'keypoints': np.array(obj['keypoints']).reshape(-1, 3).tolist(), # (x, y, visibility)
                    'skeleton': coco.cats[obj['category_id']]['skeleton']
                })
    
    return anno

def mean_value_coordinates(polygon, A): # barycentric coordinates
    """
    polygon: array of shape (n, 2)
    A: array of shape (2,)
    Returns: barycentric weights W_i
    """
    n = len(polygon)
    v = polygon - A  # vectors from A to each vertex
    d = np.linalg.norm(v, axis=1)
    angles = []

    for i in range(n):
        vi = v[i] / d[i]
        vip1 = v[(i + 1) % n] / d[(i + 1) % n]
        # angle between vi and vip1
        dot = np.clip(np.dot(vi, vip1), -1.0, 1.0)
        angles.append(np.arccos(dot))

    w = np.zeros(n)
    for i in range(n):
        prev = (i - 1) % n
        w[i] = (np.tan(angles[prev] / 2) + np.tan(angles[i] / 2)) / d[i]

    W = w / np.sum(w)
    return W

def rejection_sampling(vertices: np.ndarray): # shape (n,2)
    polygon = Polygon(vertices)
    min_x, min_y, max_x, max_y = polygon.bounds

    while True:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)
        weights = mean_value_coordinates(vertices, np.array([x, y]))
        if polygon.contains(point):
            return weights
            
def random_center(keypoints: np.ndarray) -> Dict[str,List]:
    idxs = polygons[np.random.choice(range(len(polygons)))]
    poly = keypoints[idxs]
    weights = rejection_sampling(poly)

    return {
        'polygon': idxs,
        'weights': weights.tolist(),
    }

def generate_random_list(annotations: Dict):
    random_list = []
    reference_file = 'human_face/helen/trainset/117518507_1.jpg'
    anno = annotations[reference_file][0]
    assert 'objs' in anno and len(anno['objs']) == 1
    obj = anno['objs'][0]
    keypoints = np.array(obj['keypoints']).reshape(-1, 3)[:, :2]

    for i in range(num_random_points):
        rand_poly = random_center(keypoints)
        random_list.append(rand_poly)
        # print(f"Polygon: {rand_poly['polygon']}, Weights: {rand_poly['weights']}")

    return random_list

def visualize(annotations: Dict, category: str, random_list: List[Dict]):
    for file_name, annos in annotations.items():
        if category not in file_name:
            continue
        anno = annos[0]
        for obj in anno['objs']:
            if len(obj['keypoints']) == 0:
                print(f"Empty keypoints in {file_name} for object ID {obj['id']}")
            else:
                for kp in obj['keypoints']:
                    if kp[2] not in [0, 1, 2]:
                        print(f"Invalid visibility in {file_name} for object ID {obj['id']}: {kp[2]}")
                    if kp[0] < 0 or kp[1] < 0:
                        print(f"Negative coordinates in {file_name} for object ID {obj['id']}: ({kp[0]}, {kp[1]})")

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)

                for rand_poly in random_list:
                    poly_keypoints = keypoints[rand_poly['polygon'], :2]
                    center = np.array(rand_poly['weights']) @ poly_keypoints

                    img_path = DATA_IMAGES_DIR / file_name
                    img = plt.imread(str(img_path))
                    plt.imshow(img)
                    plt.scatter(center[0], center[1], c='red', s=50, label='Random Center')
                    plt.scatter(poly_keypoints[:, 0], poly_keypoints[:, 1], c='blue', s=30, label='Polygon Keypoints')
                    plt.legend()
                    plt.title(f"{file_name} - Random Center Visualization")
                    plt.show()
                    
                break
                # poly_keypoints = poly_keypoints * weights[:, np.newaxis]
        break

def apply_weights(keypoints: np.ndarray, idxs: List[int], weights: List[float]) -> np.ndarray:
    """ Apply weights to the keypoints based on the polygon indices.
    
    Args:
        keypoints (np.ndarray): Array of keypoints of shape (n, 2).
        idxs (List[int]): Indices of the keypoints that form the polygon.
        weights (List[float]): Weights corresponding to the polygon vertices.
    
    Returns:
        mean keypoint coordinates (np.ndarray): Weighted mean of the keypoints.
    """
    keypoints = keypoints[idxs, :2]  # Select only the x, y coordinates
    if len(keypoints) != len(weights):
        raise ValueError("Number of keypoints and weights must match.")

    mean_keypoints = np.array(weights) @ keypoints
    return mean_keypoints

def validate_annotations(annotations: Dict[str, List[Dict]]):
    for file_name, annos in annotations.items():
        for i in range(len(annos)-1):
            # print(a['img_id'], a['ann_id'], a['annotation_filename'], a['annotation_name'], a['width'], a['height'])
            # print(a['objs'])
            assert annos[i]['img_id'] == annos[i+1]['img_id'], f"Image ID mismatch in {file_name}"
            assert annos[i]['ann_id'] == annos[i+1]['ann_id'], f"Annotation ID mismatch in {file_name} {annos[i]['ann_id']} vs {annos[i+1]['ann_id']}"

            # assert annotations[i]['annotation_filename'] == annotations[i+1]['annotation_filename'], f"Annotation filename mismatch in {file_name}"
            # assert annotations[i]['annotation_name'] == annotations[i+1]['annotation_name'], f"Annotation name mismatch in {file_name}"
            assert annos[i]['width'] == annos[i+1]['width'], f"Width mismatch in {file_name}"
            if len(annos[i]['objs']) != len(annos[i+1]['objs']): 
                # print(f"Number of objects mismatch in {file_name}")
                if category in file_name:
                    print(f"Number of objects mismatch in {file_name}: {len(annos[i]['objs'])} vs {len(annos[i+1]['objs'])}")
            if len(annos[i]['objs']) == len(annos[i+1]['objs']):
                for j in range(len(annos[i]['objs'])):
                    assert annos[i]['objs'][j]['id'] == annos[i+1]['objs'][j]['id'], f"Object ID mismatch in {file_name}"
                    assert annos[i]['objs'][j]['bbox'] == annos[i+1]['objs'][j]['bbox'], f"Bounding box mismatch in {file_name}"
                    assert annos[i]['objs'][j]['keypoints'] == annos[i+1]['objs'][j]['keypoints'], f"Keypoints mismatch in {file_name}"
                    assert annos[i]['objs'][j]['skeleton'] == annos[i+1]['objs'][j]['skeleton'], f"Skeleton mismatch in {file_name}"
            if category in file_name:
                for obj in annos[i]['objs']:
                    for j in range(0, len(obj['keypoints']), 3):
                        assert obj['keypoints'][j] != 0 and obj['keypoints'][j+1] != 0, f"Keypoint coordinates are zero in {file_name} for object ID {obj['id']}"

def create_annotations(annotations: Dict, category: str, random_list: List[Dict]):
    new_annotations = {}
    new_annotations['info'] = {
        'description': 'Custom annotations for human face', 
        'version': '1.0', 
        'year': '2025', 
        'contributor': 'Your Name', 
        'date_created': '2025-01-01'
    }
    new_annotations['categories'] = [
        # {"supercategory": "person", "id": 40, "name": "face", "keypoints": [], "skeleton": []}, 
        {"supercategory": "person", "id": 40, "name": "face", "keypoints": [], "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [25, 26], [27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]]},
        
    ]
    new_annotations['images'] = []
    new_annotations['annotations'] = []

    for file_name, annos in annotations.items():
        if category not in file_name:
            continue
        anno = annos[0]
        new_annotations['images'].append({
            'id': anno['img_id'],
            'file_name': file_name,
            'width': anno['width'],
            'height': anno['height'],
        })
        if len(anno['objs']) == 0:
            print(f"No objects in {file_name}")
        if len(anno['objs']) > 1:
            print(f"Multiple objects in {file_name}")
        for obj in anno['objs']:
            obj_anno = {
                'id': obj['id'],
                'image_id': anno['img_id'],
                'category_id': obj['category_id'],
                'bbox': obj['bbox'],
                'area': obj['area'],
                'iscrowd': obj['iscrowd'],
                # 'num_keypoints': len(random_list),
                # 'keypoints': []
                'num_keypoints': obj['num_keypoints'],
                'keypoints': [coord for kp in obj['keypoints'] for coord in kp],  # flatten to [x,y,v,...]
            }
            # keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            # for rand in random_list:
            #     obj_anno['keypoints'].extend(apply_weights(keypoints, rand['polygon'], rand['weights']).tolist())
            #     obj_anno['keypoints'].append(1)  # visibility

            new_annotations['annotations'].append(obj_anno)
    
    return new_annotations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create face annotations")
    parser.add_argument('--random', action='store_true', help="Generate random keypoints")
    parser.add_argument('--create', action='store_true', help="Create face annotations")
    parser.add_argument('--visualize', action='store_true', help="Visualize annotations")
    parser.add_argument('--random_file', type=str, default='data/human_face_random_keypoints.json', help="File to save random keypoints")
    parser.add_argument('--annotations_file', type=str, default='data/human_face_annotations.json', help="File to save annotations")
    args = parser.parse_args()

    annotations = load_annotations(DATA_ANNOTATIONS_DIR)
    category = "human_face"
    random_file = Path(args.random_file)
    annotations__file = Path(args.annotations_file)

    if args.random:
        random_list = generate_random_list(annotations)
        with open(random_file, 'w') as f: 
            json.dump(random_list, f, indent=4)

    if args.create:
        with open(random_file, 'r') as f:
            random_list = json.load(f)
        new_annotations = create_annotations(annotations, category, random_list)
        with open(annotations__file, 'w') as f:
            json.dump(new_annotations, f, indent=4)

    #%%

    # {"keypoints": [136.0, 148.0, 2.0, 189.0, 78.0, 2.0, 199.0, 155.0, 2.0, 181.0, 130.0, 2.0, 128.0, 83.0, 2.0, 158.0, 76.0, 2.0, 0.0, 0.0, 0.0, 205.0, 143.0, 2.0, 0.0, 0.0, 0.0, 132.0, 118.0, 2.0, 148.0, 84.0, 2.0, 232.0, 178.0, 2.0, 159.0, 168.0, 2.0, 205.0, 283.0, 2.0, 160.0, 103.0, 2.0], "image_id": 1400000000011055, "id": 1400000001011055, "num_keypoints": 13, "bbox": [102.0, 63.0, 163.0, 277.0], "iscrowd": 0, "area": 45151.0, "category_id": 14}


#%%
# import json

# with open("/volume/datasets/mp100/annotations/mp100_split1_test.json", "r") as f:
#     annotations = json.load(f)

# print(annotations.keys())

# #%%

# print(annotations['info'])
# print(annotations['images'][0].keys())
# print(annotations['annotations'][0].keys())
# print(json.dumps(annotations['categories'], indent=4))