import os
import sys
import cv2
import numpy as np

def denorm(x):
    xmax, xmin = x.max(), x.min()
    return (x - xmin) / (xmax - xmin)

fname = sys.argv[1] # "samples_1000x128x128x3.npz"
data = np.load(fname)
path = fname[:-len(".npz")]
os.makedirs(path, exist_ok=True)
for i in range(data["arr_0"].shape[0]):
    img = data["arr_0"][i]
    img_name = "img_%d_label%d.png" % (i, data["arr_1"][i])
    cv2.imwrite(os.path.join(path, img_name), img)
    print(path, img_name, img.shape)
