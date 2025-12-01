import torch
import numpy as np
from scipy.spatial import Delaunay
import os
import pickle

class RefineHead:
    def __init__(self):
        pass

    def __call__(self, 
                 target_s,
                 preds,
                 coarse_s,
                 mask_s):

        B, n, K, c = target_s.shape

        assert c == 2, "Only support 2D keypoints now."
        assert n == 1, "Only support one shot now."

        # Retrieve support keypoints coordinates from heatmap target_s (B, K, H, W) of n shots (target_s is a list of (n, B, K, H, W)) by averaging
        # if use_heatmap:
        #     B, K, H, W = target_s[0].shape
        #     n = len(target_s)
        #     target_s = torch.stack(target_s, dim=0).view(n, B, K, H, W)
        #     # Normalize heatmaps
        #     norm = target_s.view(n, B, K, -1).sum(dim=-1, keepdim=True) + 1e-8
        #     target_s_norm = target_s / norm.view(n, B, K, 1, 1)
        #     # Create coordinate grids
        #     y_grid = torch.arange(H, device=target_s.device).view(1, 1, 1, H, 1).float()
        #     x_grid = torch.arange(W, device=target_s.device).view(1, 1, 1, 1, W).float()
        #     # Weighted average for coordinates
        #     x = (target_s_norm * x_grid).sum(dim=(-2, -1))
        #     y = (target_s_norm * y_grid).sum(dim=(-2, -1))
        #     target_s = torch.stack((x, y), dim=-1)  # (n, B, K, 2)
        #     # target_s = target_s.mean(dim=0)  # (B, K, 2)
        #     # target_s = target_s.view(B, K, 2)
        # else:
        #     B, K = target_s[0].shape[:2]
        #     n = len(target_s)
        
        batch_with_fine_idx = torch.where((mask_s * (1 - coarse_s)).sum(dim=1) > 0)[0].tolist()
        if len(batch_with_fine_idx) == 0:
            return preds

        results = []
        preds_np = preds[-1].cpu().numpy().copy()  # (B, K, 2)
        target_s_np = target_s.squeeze(1).cpu().numpy()
        mask_s_np = mask_s.cpu().numpy()  # (B, K), 1 for visible, 0 for invisible
        coarse_s_np = coarse_s.cpu().numpy()  # (B, K), 1 for coarse, 0 for fine


        for b in batch_with_fine_idx:
            # Get visible keypoints indices
            visible_anchor_idx = np.where(mask_s_np[b] * coarse_s_np[b] > 0)[0]
            # assert len(visible_anchor_idx) >= 3, "Need at least 3 visible keypoints for Delaunay triangulation."
            # Delaunay triangulation on visible keypoints in preds
            pts = target_s_np[b][visible_anchor_idx]  # (V, 2)
            assert len(np.unique(pts, axis=0)) >= 3, "Need at least 3 unique visible keypoints for Delaunay triangulation."
            tri = Delaunay(pts)
            delaunay_indices = np.unique(tri.simplices)
            delaunay_pts = pts[delaunay_indices]

            # print(f"{pts = }")
            # print(f"{tri.simplices = }")

            visible_fine_idx = np.where(mask_s_np[b] * (1 - coarse_s_np[b]) > 0)[0] # Indices of target keypoints to be refined
            mapped = []
            for t in target_s_np[b][visible_fine_idx]:  # (2,)
                simplex = tri.find_simplex(t)
                if simplex == -1:
                    # Outside convex hull, find nearest triangle
                    # print("Warning: target keypoint outside convex hull of visible keypoints.")
                    dists = np.linalg.norm(delaunay_pts - t, axis=1)
                    # print(f"{dists = }")
                    nearest_vertex = np.argmin(dists)
                    neighboring_simplices = np.where(tri.simplices == delaunay_indices[nearest_vertex])[0]
                    # print(f"{nearest_vertex = }, {neighboring_simplices = }")
                    simplex = neighboring_simplices[0]

                vertices = tri.simplices[simplex]
                tri_pts = pts[vertices]  # (3, 2)
                # Compute barycentric coordinates
                T = np.hstack((tri_pts, np.ones((3, 1))))
                v = np.append(t, 1)
                bary = np.linalg.solve(T.T, v)
                # Map barycentric coordinates to query prediction
                mapped_pt = np.dot(bary, preds_np[b][visible_anchor_idx][vertices])
                mapped.append(mapped_pt)
            preds_np[b][visible_fine_idx] = np.array(mapped)

            results.append(preds_np[b])

        results = torch.tensor(np.stack(results, axis=0), dtype=preds.dtype, device=preds.device)  # (B, K, 2)

        return torch.vstack((preds, results.unsqueeze(0)))
