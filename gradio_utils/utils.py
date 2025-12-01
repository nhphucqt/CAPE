import io
import copy
import json
import random
import collections
import gradio as gr
import numpy as np
import psutil
import torch
from PIL import ImageDraw, Image, ImageEnhance, ImageFont
from matplotlib import pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmpose.core import wrap_fp16_model
from mmpose.models import build_posenet
from torchvision import transforms
import matplotlib.patheffects as mpe
from pathlib import Path
from datetime import datetime
import cv2

from EdgeCape import TopDownGenerateTargetFewShot
from demo import Resize_Pad
from EdgeCape.models import *

COLORS = [(216, 194, 107), (66, 130, 103), (200, 77, 122), (149, 232, 129), (72, 193, 158), (64, 232, 251), (207, 230, 79), (186, 230, 175), (120, 25, 111), (156, 233, 247), (122, 221, 66), (206, 140, 3), (184, 102, 211), (171, 0, 126), (222, 62, 83), (222, 48, 145), (61, 247, 205), (114, 20, 81), (130, 238, 27), (141, 180, 140), (208, 138, 246), (154, 150, 113), (152, 98, 147), (74, 48, 47), (156, 168, 121), (22, 193, 224), (236, 215, 229), (236, 138, 100), (180, 70, 207), (217, 229, 150), (243, 148, 115), (169, 255, 234), (203, 21, 156), (124, 161, 216), (62, 187, 29), (56, 203, 85), (208, 25, 37), (178, 11, 146), (232, 136, 174), (6, 162, 155), (147, 100, 94), (251, 9, 5), (246, 47, 31), (53, 204, 239), (5, 108, 25), (66, 56, 165), (89, 46, 128), (10, 25, 252), (51, 91, 187), (214, 235, 43), (172, 247, 14), (173, 216, 87), (64, 152, 113), (44, 120, 104), (145, 130, 79), (91, 214, 64), (143, 3, 189), (85, 11, 71), (61, 244, 90), (73, 91, 242), (162, 158, 183), (99, 106, 166), (0, 49, 85), (61, 163, 96), (224, 145, 106), (102, 179, 107), (169, 11, 114), (66, 40, 135), (124, 143, 193), (226, 126, 79), (119, 207, 224), (207, 48, 255), (162, 21, 185), (252, 102, 173), (80, 54, 183), (0, 210, 135), (25, 30, 166), (223, 71, 250), (25, 218, 101), (20, 70, 115), (202, 220, 34), (133, 166, 88), (223, 71, 4), (10, 174, 142), (242, 240, 232), (10, 191, 179), (167, 182, 231), (163, 95, 137), (53, 150, 2), (38, 85, 202), (183, 86, 158), (10, 41, 251), (74, 101, 140), (75, 122, 61), (12, 45, 133), (18, 103, 84), (106, 25, 232), (121, 215, 249), (87, 122, 179), (109, 77, 188)]


def process_img(support_image, global_state):
    global_state['images']['image_orig'] = support_image
    global_state['images']['image_kp'] = support_image
    global_state['images']['image_skeleton'] = support_image
    global_state['curr_type_point'] = "start"
    global_state['prev_point'] = None
    global_state['bbox'] = []
    kp_image = support_image
    skel_image = support_image
    if global_state["load_example"]:
        global_state["load_example"] = False
        return global_state['images']['image_kp'], global_state
    _, _, _ = reset_kp(global_state)
    return kp_image, skel_image, global_state

def process_query_img(query_image, global_state):
    global_state['images']['query_image_orig'] = query_image
    _ = reset_bbox(global_state)
    return global_state


def adj_mx_from_edges(num_pts, skeleton, device='cuda', normalization_fix=True):
    adj_mx = torch.empty(0, device=device)
    batch_size = len(skeleton)
    for b in range(batch_size):
        edges = torch.tensor(skeleton[b]).long()
        adj = torch.zeros(num_pts, num_pts, device=device)
        adj[edges[:, 0], edges[:, 1]] = 1
        adj_mx = torch.concatenate((adj_mx, adj.unsqueeze(0)), dim=0)
    trans_adj_mx = torch.transpose(adj_mx, 1, 2)
    cond = (trans_adj_mx > adj_mx).float()
    adj = adj_mx + trans_adj_mx * cond - adj_mx * cond
    return adj

def pad_to_square(img, pad_value=0):
    """Pad a (H, W, C) or (H, W) image to a square shape."""
    h, w = img.shape[:2]
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padding = ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w))
    if img.ndim == 3:
        padding += ((0, 0),)
    return np.pad(img, padding, mode='constant', constant_values=pad_value)

def plot_results(support_img, query_img, support_kp, support_w, query_kp, query_w,
                 skeleton=None, prediction=None, radius=6, in_color=None,
                 original_skeleton=None, img_alpha=0.6, target_keypoints=None, coarse=None):
    ori_skeleton = copy.deepcopy(original_skeleton)
    max_side = 512
    # resize query image to 512
    h, w, c = query_img.shape
    scale = max_side / max(h, w)
    query_img = cv2.resize(query_img, (int(w * scale), int(h * scale)))
    query_img = pad_to_square(query_img, pad_value=0)
    h, w, c = query_img.shape
    # prediction = prediction[-1, :] * np.array([w, h]) # something wrong here
    prediction = prediction[-1, :] * h
    if coarse is None:
        coarse = np.full(prediction.shape[0], 1.0)
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(original_skeleton, list):
        original_skeleton = adj_mx_from_edges(num_pts=100, skeleton=[skeleton]).cpu().numpy()[0]
    # query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))
    img = query_img
    w = query_w
    keypoint = prediction
    adj = skeleton
    color = None
    f, axes = plt.subplots()
    axes.imshow(img)
    for k in range(keypoint.shape[0]):
        if w[k] > 0:
            kp = keypoint[k, :2]
            # c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
            c = (1, 0, 0, 0.75) if coarse[k] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp,
                               radius,
                               color=c,
                               path_effects=[mpe.withStroke(linewidth=2, foreground='black')],
                               zorder=200)
            axes.add_patch(patch)
            axes.text(kp[0], kp[1], k, fontsize=(radius+4), color='white', ha="center", va="center",
                      zorder=300,
                      path_effects=[
                          mpe.withStroke(linewidth=max(1, int((radius+4) / 5)), foreground='black')])
            plt.draw()

    if adj is not None:
        # c = (1, 0, 0, 0.75)
        c = tuple(random.choices(range(256), k=3)) + (200,)
        max_skel_val = np.max(adj)
        draw_skeleton = adj / max_skel_val * 3
        # for i in range(1, keypoint.shape[0]):
        #     for j in range(0, i):
        #         if w[i] > 0 and w[j] > 0 and original_skeleton[i][j] > 0:
        #             if color is None:
        #                 num_colors = int((adj > 0.05).sum() / 2)
        #                 color = iter(plt.cm.rainbow(np.linspace(0, 1, num_colors + 1)))
        #                 c = next(color)
        #             elif isinstance(color, str):
        #                 c = color
        #             elif isinstance(color, collections.Iterable):
        #                 c = next(color)
        #             else:
        #                 raise ValueError("Color must be a string or an iterable")
        #         if w[i] > 0 and w[j] > 0 and adj[i][j] > 0:
        #             width = draw_skeleton[i][j]
        #             stroke_width = width + (width / 3)
        #             patch = plt.Line2D([keypoint[i, 0], keypoint[j, 0]],
        #                                [keypoint[i, 1], keypoint[j, 1]],
        #                                linewidth=width, color=c, alpha=0.6,
        #                                path_effects=[mpe.withStroke(linewidth=stroke_width, foreground='black')],
        #                                zorder=1)
        #             axes.add_artist(patch)
        color = iter(COLORS)
        for sk in ori_skeleton:
            i, j = sk
            width = draw_skeleton[i][j]
            # stroke_width = width + (width / 3)
            stroke_width = 0
            c = next(color) + (200,)
            c = tuple([x / 255.0 for x in c])
            patch = plt.Line2D([keypoint[i, 0], keypoint[j, 0]],
                                [keypoint[i, 1], keypoint[j, 1]],
                                linewidth=width, color=c,
                                path_effects=[mpe.withStroke(linewidth=stroke_width, foreground='black')],
                                zorder=1)
            axes.add_artist(patch)

        plt.axis('off')  # command for hiding the axis.
        print(f"{img.shape = }")
        axes.set_xlim(0, img.shape[1])
        axes.set_ylim(img.shape[0], 0)
        # f.subplots_adjust(left=0, right=1, top=1, bottom=0)
        f.tight_layout(pad=0)
        # plt.margins(0, 0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        plt.close(f)
        buf.seek(0)
        return Image.open(buf)

def plot_additional_results(support_img, query_img, support_kp, support_w, query_kp, query_w,
                            skeleton=None, prediction=None, radius=6, in_color=None,
                            original_skeleton=None, img_alpha=0.6, target_keypoints=None,
                            add_data={}):
    h, w, c = support_img.shape
    prediction = prediction[-1] * h
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(original_skeleton, list):
        original_skeleton = adj_mx_from_edges(num_pts=100, skeleton=[skeleton]).cpu().numpy()[0]
    query_img = (query_img - np.min(query_img)) / (np.max(query_img) - np.min(query_img))
    img = query_img
    w = query_w
    keypoint = prediction
    adj = skeleton
    color = None
    plt.close('all')
    f, axes = plt.subplots()
    axes.imshow(img, alpha=img_alpha)
    for k in range(keypoint.shape[0]):
        if w[k] > 0:
            kp = keypoint[k, :2]
            c = (1, 0, 0, 0.75) if w[k] == 1 else (0, 0, 1, 0.6)
            patch = plt.Circle(kp,
                               radius,
                               color=c,
                               path_effects=[mpe.withStroke(linewidth=2, foreground='black')],
                               zorder=200)
            axes.add_patch(patch)
            axes.text(kp[0], kp[1], k, fontsize=(radius + 4), color='white', ha="center", va="center",
                      zorder=300,
                      path_effects=[
                          mpe.withStroke(linewidth=max(1, int((radius + 4) / 5)), foreground='black')])
        
            if add_data.get('query_bbox', None) is not None:
                # Draw bbox
                bbox = add_data['query_bbox']
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1],
                                     linewidth=2,
                                     edgecolor='blue',
                                     facecolor='none',
                                     zorder=100)
                axes.add_patch(rect)
                # Draw PCK circles for 0.05, 0.1, 0.2
                bbox_width = add_data['query_bbox'][2] - add_data['query_bbox'][0]
                bbox_height = add_data['query_bbox'][3] - add_data['query_bbox'][1]
                for pck, color, zorder in zip([0.05, 0.1, 0.2], ['orange', 'yellow', 'gray'], [152, 151, 150]):
                    pck_radius = pck * max(bbox_width, bbox_height)
                    axes.add_patch(plt.Circle(
                        kp,
                        pck_radius,
                        fill=False,
                        linestyle='--',
                        edgecolor=color,
                        linewidth=1,
                        zorder=zorder
                    ))
            plt.draw()

    if adj is not None:
        max_skel_val = np.max(adj)
        draw_skeleton = adj / max_skel_val * 6
        for i in range(1, keypoint.shape[0]):
            for j in range(0, i):
                if w[i] > 0 and w[j] > 0 and original_skeleton[i][j] > 0:
                    if color is None:
                        num_colors = int((adj > 0.05).sum() / 2)
                        color = iter(plt.cm.rainbow(np.linspace(0, 1, num_colors + 1)))
                        c = next(color)
                    elif isinstance(color, str):
                        c = color
                    elif isinstance(color, collections.Iterable):
                        c = next(color)
                    else:
                        raise ValueError("Color must be a string or an iterable")
                if w[i] > 0 and w[j] > 0 and adj[i][j] > 0:
                    width = draw_skeleton[i][j]
                    stroke_width = width + (width / 3)
                    patch = plt.Line2D([keypoint[i, 0], keypoint[j, 0]],
                                       [keypoint[i, 1], keypoint[j, 1]],
                                       linewidth=width, color=c, alpha=0.6,
                                       path_effects=[mpe.withStroke(linewidth=stroke_width, foreground='black')],
                                       zorder=1)
                    axes.add_artist(patch)

        plt.axis('off')  # command for hiding the axis.
        return f

def process(query_img, state,
            models_list, model_name):
    cfg_path = models_list[model_name]['config']
    checkpoint_path = models_list[model_name]['checkpoint']
    print(state)
    query_img = state['images']['query_image_orig']
    device = print_memory_usage()
    cfg = Config.fromfile(cfg_path)
    width, height, _ = np.array(state['images']['image_orig']).shape
    kp_src_np = np.array(state['points']).copy().astype(np.float32)
    kp_src_np[:, 0] = kp_src_np[:, 0] / width * 256
    kp_src_np[:, 1] = kp_src_np[:, 1] / height * 256
    kp_src_np = kp_src_np.copy()
    kp_src_tensor = torch.tensor(kp_src_np).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Resize_Pad(256, 256)
    ])

    # if len(state['skeleton']) == 0:
    #     state['skeleton'] = [(0, 0)]

    support_img = preprocess(state['images']['image_orig']).flip(0)[None]
    np_query = np.array(query_img)[:, :, ::-1].copy()
    q_img = preprocess(np_query).flip(0)[None]
    # Resize bbox coordinates from original image size to 256x256
    query_bbox = None
    print(f"{type(state) = }")
    print(f"{type(state['bbox']) = }")
    if len(state['bbox']) >= 2:
        query_bbox = state['bbox'][-2:]
        orig_h, orig_w = np_query.shape[:2]
        # Calculate scale and padding for Resize_Pad(256, 256)
        scale = min(256 / orig_w, 256 / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_w = (256 - new_w) // 2
        pad_h = (256 - new_h) // 2

        print(f"{query_bbox = }")
        # Clamp bbox to image bounds
        query_bbox = [query_bbox[0][0], query_bbox[0][1], query_bbox[1][0], query_bbox[1][1]]
        query_bbox = [
            max(0, min(query_bbox[0], query_bbox[2])),
            max(0, min(query_bbox[1], query_bbox[3])),
            min(orig_w, max(query_bbox[0], query_bbox[2])),
            min(orig_h, max(query_bbox[1], query_bbox[3])),
        ]
        # Scale bbox
        query_bbox = [
            int(query_bbox[0] * scale) + pad_w,
            int(query_bbox[1] * scale) + pad_h,
            int(query_bbox[2] * scale) + pad_w,
            int(query_bbox[3] * scale) + pad_h
        ]
    # Create heatmap from keypoints
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array([256, 256])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.cat(
        (kp_src_tensor, torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.cat(
        (torch.ones_like(kp_src_tensor),
         torch.zeros(kp_src_tensor.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(data_cfg,
                                                                 kp_src_3d,
                                                                 kp_src_3d_weight,
                                                                 sigma=1)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.ones_like(
        torch.tensor(target_weight_s).float()[None])
    target_coarse_s = torch.tensor(np.array(state['coarse'])).float()[None, :, None]

    print(f"{target_s.shape = }")
    print(f"{target_weight_s.shape = }")
    print(f"{target_coarse_s.shape = }")

    data = {
        'img_s': [support_img.to(device)],
        'img_q': q_img.to(device),
        'target_s': [target_s.to(device)],
        'target_weight_s': [target_weight_s.to(device)],
        'target_coarse_s': [target_coarse_s.to(device)],
        'target_q': None,
        'target_weight_q': None,
        'return_loss': False,
        'img_metas': [{'sample_skeleton': [state['skeleton']],
                       'query_skeleton': state['skeleton'],
                       'sample_joints_3d': [kp_src_3d.cpu().numpy()],
                       'query_joints_3d': kp_src_3d.cpu().numpy(),
                       'sample_center': [kp_src_tensor.mean(dim=0).cpu().numpy()],
                       'query_center': kp_src_tensor.mean(dim=0).cpu().numpy(),
                       'sample_scale': [
                           (kp_src_tensor.max(dim=0)[0] -
                           kp_src_tensor.min(dim=0)[0]).cpu().numpy()
                       ],
                       'query_scale': (kp_src_tensor.max(dim=0)[0] -
                                      kp_src_tensor.min(dim=0)[0]).cpu().numpy(),
                       'sample_rotation': [0],
                       'query_rotation': 0,
                       'sample_bbox_score': [1],
                       'query_bbox_score': 1,
                       'query_image_file': '',
                       'sample_image_file': [''],
                       }]
    }
    # Load model
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.eval().to(device)
    with torch.no_grad():
        outputs = model(**data)
    # visualize results
    vis_s_weight = target_weight_s[0]
    vis_s_image = support_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    vis_q_image = q_img[0].detach().cpu().numpy().transpose(1, 2, 0)

    # print(vis_q_image.shape, vis_s_image.shape)

    # print(f"{outputs['skeleton'][1] = }")
    # print(f"{outputs['skeleton'][1].shape = }")

    print(f"{vis_s_image.shape = }")
    print(f"{vis_q_image.shape = }")

    num_kp = vis_s_weight.shape[0]
    adj = np.zeros((num_kp, num_kp))
    for sk in state['skeleton']:
        adj[sk[0], sk[1]] = 0.2
        adj[sk[1], sk[0]] = 0.2

    support_kp = kp_src_3d
    out = plot_results(vis_s_image,
                       np.array(query_img).copy(),
                       support_kp,
                       vis_s_weight,
                       None,
                       vis_s_weight,
                    #    outputs['skeleton'][1],
                       adj,
                       torch.tensor(outputs['points']).squeeze(),
                       original_skeleton=state['skeleton'],
                       img_alpha=1.0,
                       radius=9,
                       coarse=np.array(state['coarse']),
                       )

    additional_data = {
        'query_bbox': query_bbox,
    }

    add_out = plot_additional_results(vis_s_image,
                                      vis_q_image,
                                      support_kp,
                                      vis_s_weight,
                                      None,
                                      vis_s_weight,
                                    #   outputs['skeleton'][1],
                                      adj,
                                      torch.tensor(outputs['points']).squeeze(),
                                      original_skeleton=state['skeleton'],
                                      img_alpha=1.0,
                                      add_data=additional_data,
                                      )
    return out, add_out


def update_examples(support_img, query_image, global_state_str):
    example_state = json.loads(global_state_str)
    example_state["load_example"] = True
    example_state["curr_type_point"] = "start"
    example_state["prev_point"] = None
    example_state['images'] = {}
    example_state['images']['image_orig'] = support_img
    example_state['images']['image_kp'] = support_img
    example_state['images']['image_skeleton'] = support_img
    example_state['images']['query_image_orig'] = query_image
    example_state['images']['query_image'] = query_image
    example_state['bbox'] = []
    image_draw = example_state['images']['image_orig'].copy()
    pts_list = example_state['points']
    colors = iter(COLORS)
    for limb in example_state['skeleton']:
        prev_point = pts_list[limb[0]]
        curr_point = pts_list[limb[1]]
        points = [prev_point, curr_point]
        c = next(colors) + (200, )
        image_draw = draw_limbs_on_image(image_draw,
                                         points,
                                         color=c
                                         )
    for idx, (xy, coarse) in enumerate(zip(example_state['points'], example_state['coarse'])):
        image_draw = update_image_draw(
            image_draw,
            xy,
            example_state,
            is_coarse=(coarse == 1.0),
            draw_text=idx
        )
    skel_image = image_draw.copy()
    example_state['images']['image_skel'] = skel_image
    image_draw = example_state['images']['image_orig'].copy()
    for idx, (xy, coarse) in enumerate(zip(example_state['points'], example_state['coarse'])):
        image_draw = update_image_draw(
            image_draw,
            xy,
            example_state,
            is_coarse=(coarse == 1.0),
            draw_text=idx
        )
    kp_image = image_draw.copy()
    example_state['images']['image_kp'] = kp_image
    return (support_img,
            kp_image,
            skel_image,
            query_image,
            example_state)


def get_select_coords(is_coarse, global_state,
                      evt: gr.SelectData
                      ):
    """This function only support click for point selection
    """
    xy = evt.index
    global_state["points"].append(xy)
    global_state["coarse"].append(1.0 if is_coarse else 0.0)
    image_raw = global_state['images']['image_kp']
    image_draw = update_image_draw(
        image_raw,
        xy,
        global_state,
        is_coarse=is_coarse,
        draw_text=len(global_state["points"])-1
    )
    global_state['images']['image_kp'] = image_draw
    return global_state, image_draw

def get_select_bbox_coords(query_image, global_state,
                      evt: gr.SelectData
                      ):
    """This function only support click for point selection
    """
    xy = evt.index
    global_state["bbox"].append(xy)
    image_raw = global_state['images']['query_image_orig']
    if len(global_state["bbox"]) >= 2:
        p1 = global_state["bbox"][-1]
        p2 = global_state["bbox"][-2]
        bbox = [min(p1[0], p2[0]), min(p1[1], p2[1]),
                max(p1[0], p2[0]), max(p1[1], p2[1])]
        image_draw = update_image_draw(image_raw, p1, global_state)
        image_draw = update_image_draw(image_draw, p2, global_state)
        draw = ImageDraw.Draw(image_draw)
        draw.rectangle(bbox, outline="blue", width=3)
    else:
        image_draw = update_image_draw(image_raw, xy, global_state)
    global_state['images']['query_image'] = image_draw
    return global_state, image_draw

def get_closest_point_idx(pts_list, xy):
    x, y = xy
    closest_point = min(pts_list, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    closest_point_index = pts_list.index(closest_point)
    return closest_point_index

def confirm_kp(global_state):
    image = global_state["images"]["image_kp"]
    image_draw = image.copy()
    for limb in global_state["skeleton"]:
        prev_point = global_state["points"][limb[0]]
        curr_point = global_state["points"][limb[1]]
        points = [prev_point, curr_point]
        image_draw = draw_limbs_on_image(image_draw,
                                    points
                                    )  
    global_state["images"]["image_skel"] = image_draw
    # global_state["skeleton"] = []
    # global_state["curr_type_point"] = "start"
    # global_state["prev_point"] = None
    return image_draw, global_state

def reset_skeleton(global_state):
    image = global_state["images"]["image_kp"]
    global_state["images"]["image_skel"] = image
    global_state["skeleton"] = []
    global_state["curr_type_point"] = "start"
    global_state["prev_point"] = None
    return image, global_state

def toggle_coarse_fine(is_coarse):
    print(f"{is_coarse = }")


def reset_kp(global_state):
    image = global_state["images"]["image_orig"]
    global_state["images"]["image_kp"] = image
    global_state["images"]["image_skel"] = image
    global_state["skeleton"] = []
    global_state["points"] = []
    global_state["coarse"] = []
    global_state["curr_type_point"] = "start"
    global_state["prev_point"] = None
    return image, image, global_state

def reset_bbox(global_state):
    image = global_state["images"]["query_image_orig"]
    global_state["images"]["query_image"] = image
    global_state["bbox"] = []
    return image, global_state

def select_skeleton(global_state,
                    evt: gr.SelectData,
                    ):
    xy = evt.index
    pts_list = global_state["points"]
    closest_point_idx = get_closest_point_idx(pts_list, xy)
    image_raw = global_state['images']['image_skel']
    if global_state["curr_type_point"] == "end":
        prev_point_idx = global_state["prev_point_idx"]
        prev_point = pts_list[prev_point_idx]
        points = [prev_point, xy]
        image_draw = draw_limbs_on_image(image_raw,
                                         points
                                         )
        global_state['images']['image_skel'] = image_draw
        global_state['skeleton'].append([prev_point_idx, closest_point_idx])
        global_state["curr_type_point"] = "start"
        global_state["prev_point_idx"] = None
    else:
        global_state["prev_point_idx"] = closest_point_idx
        global_state["curr_type_point"] = "end"
    return global_state, global_state['images']['image_skel']


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def update_image_draw(image, points, global_state, is_coarse=True, draw_text=None):
    if len(global_state["points"]) < 2:
        alpha = 0.5
    else:
        alpha = 1.0
    image_draw = draw_points_on_image(image, points, alpha=alpha, is_coarse=is_coarse, draw_text=draw_text)
    return image_draw


def print_memory_usage():
    # Print system memory usage
    print(f"System memory usage: {psutil.virtual_memory().percent}%")

    # Print GPU memory usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9} GB")
        print(
            f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1e9} GB")
        device_properties = torch.cuda.get_device_properties(device)
        available_memory = device_properties.total_memory - \
                           torch.cuda.max_memory_allocated()
        print(f"Available GPU memory: {available_memory / 1e9} GB")
    else:
        device = "cpu"
        print("No GPU available")
    return device

def draw_limbs_on_image(image,
                        points,
                        width_scale=0.02,
                        color=None):
    if color is None:
        color = tuple(random.choices(range(256), k=3)) + (200,)
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    p_start, p_target = points
    rad_draw = int(max(image.size[0], image.size[1]) * width_scale) / 2
    if p_start is not None and p_target is not None:
        p_draw = int(p_start[0]), int(p_start[1])
        t_draw = int(p_target[0]), int(p_target[1])
        overlay_draw.line(
            (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
            fill=color,
            width=int(rad_draw),
        )

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_points_on_image(image,
                         points,
                         radius_scale=0.02,
                         alpha=1.,
                         is_coarse=True,
                         draw_text=None):
    if alpha < 1:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    if is_coarse:
        p_color = (255, 0, 0)
    else:
        p_color = (0, 0, 255)
    rad_draw = int(max(image.size[0], image.size[1]) * radius_scale)
    if points is not None:
        p_draw = int(points[0]), int(points[1])
        overlay_draw.ellipse(
            (
                p_draw[0] - rad_draw,
                p_draw[1] - rad_draw,
                p_draw[0] + rad_draw,
                p_draw[1] + rad_draw,
            ),
            fill=p_color,
            )
        if draw_text is not None:
            text = str(draw_text)
            font = ImageFont.truetype("/volume/DejaVuSans.ttf", size=rad_draw*2 if len(text)==1 else rad_draw*1.75)
            overlay_draw.text(
                (p_draw[0]-(rad_draw*2/3 if len(text)==1 else rad_draw*3/4), p_draw[1] - rad_draw*5/4),
                str(draw_text),
                font=font,
                fill=(255, 255, 255),
                align="center",
                stroke_width=rad_draw//5,
                stroke_fill="black",
            )

    return Image.alpha_composite(image.convert("RGBA"), overlay_rgba).convert("RGB")

def select_model(config_name, global_state):
    print(f"Selected model: {config_name}")
    # global_state['model_name'] = config_name
    # return global_state

def save_support_query(examples_list, global_state):
    tmp_folder = Path("tmp/gradio/saved")
    tmp_folder.mkdir(parents=True, exist_ok=True)

    support_image = global_state['images']['image_orig']
    query_image = global_state['images']['query_image_orig']

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    support_image_path = tmp_folder / f"support_image_{dt}.png"
    query_image_path = tmp_folder / f"query_image_{dt}.png"

    support_image.save(support_image_path)
    query_image.save(query_image_path)

    keypoints = global_state['points']
    coarse = global_state['coarse']
    skeleton = global_state['skeleton']    

    record = {
        "support_image": str(support_image_path),
        "query_image": str(query_image_path),
        "points": keypoints,
        "coarse": coarse,
        "skeleton": skeleton,
    }
    with open(tmp_folder / "saved_state.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    print("Saved support and query images along with keypoints.")

    examples_list.append([
        str(support_image_path),
        str(query_image_path),
        json.dumps({
            'points': keypoints,
            'coarse': coarse,
            'skeleton': skeleton,
        })
    ])

    return gr.Dataset.update(samples=examples_list), examples_list