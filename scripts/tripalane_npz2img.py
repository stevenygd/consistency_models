import os
import sys
import cv2
import numpy as np


def denorm(x):
    xmax, xmin = x.max(), x.min()
    return (x - xmin) / (xmax - xmin)


idx = random.choice(np.arange(data["arr_0"].shape[0]))
triplane = data["arr_0"][idx]
for i in range(triplane.shape[-1]):
    triplane_slice = (denorm(triplane[..., i]) * 255).astype(np.uint8)
    outf = "results/tri_planes/10k/generate-idx%d-dim%d.jpg"%(idx, i)
    print(outf, triplane_slice.shape)
    os.makedirs(os.path.dirname(outf), exist_ok=True)
    cv2.imwrite(outf, triplane_slice)
    print(triplane_slice.max(), triplane_slice.min())
