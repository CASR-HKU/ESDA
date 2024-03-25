import numpy as np
import torch


def draw_gaussian(pt, width, height, gaussian=1, dist=3):
    hm_img = np.zeros((height, width))
    ul = [int(pt[0] - dist), int(pt[1] - dist)]
    br = [int(pt[0] + dist + 1), int(pt[1] + dist + 1)]
    size = 2 * dist + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = gaussian
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * 10

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], width)
    img_y = max(0, ul[1]), min(br[1], height)
    try:
        hm_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    except:
        a = 1
    return torch.from_numpy(hm_img)


def generate_heatmap_label(pts, w, h):
    temporal_length = pts.shape[0]
    target_heatmap = np.zeros((temporal_length, w, h))
    for idx, pt in enumerate(pts):
        target_heatmap[idx] = draw_gaussian(pt, h, w)
    return target_heatmap
