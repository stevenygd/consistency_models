import os
import sys
import cv2
import numpy as np


def denorm(x):
    xmax, xmin = x.max(), x.min()
    return (x - xmin) / (xmax - xmin)

data = np.load(sys.argv[1], allow_pickle=True)
outdir = sys.argv[2]
os.makedirs(os.path.dirname(outdir), exist_ok=True)

triplanes = np.arange(data["arr_0"])
for idx in range(triplanes.shape[0]):
    triplane = triplanes[idx]
    outf = os.path.join(outdir, "triplane-idx%d.npy" % idx)
    print(outf, triplane.shape, triplane.max(), triplane.min())
    cv2.imwrite(outf, triplane)

# idx = random.choice(np.arange(data["arr_0"].shape[0]))
# triplane = data["arr_0"][idx]
# for i in range(triplane.shape[-1]):
    # triplane_slice = (denorm(triplane[..., i]) * 255).astype(np.uint8)
    # outf = "results/tri_planes/10k/generate-idx%d-dim%d.jpg"%(idx, i)
    # print(outf, triplane_slice.shape)
    # os.makedirs(os.path.dirname(outf), exist_ok=True)
    # cv2.imwrite(outf, triplane_slice)
    # print(triplane_slice.max(), triplane_slice.min())
