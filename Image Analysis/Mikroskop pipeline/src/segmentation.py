import numpy as np
from cellSAM import segment_cellular_image
img = np.load("sample_imgs/yeaz.npy")
mask, _, _ = segment_cellular_image(img, device='cuda')