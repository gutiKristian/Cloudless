import glob

import rasterio
import numpy as np
from s2cloudless import S2PixelCloudDetector
from Pipeline.Plotting import Plot
import skimage.transform

# bands we are going to work with
bn = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
path = "/home/xgutic/Desktop/S2B_MSIL1C_20211218T095319_N0301_R079_T33UXQ_20211218T103830.SAFE/GRANULE/L1C_T33UXQ_A024987_20211218T095349/IMG_DATA/"
b_path = [glob.glob(path + f"*{b}*")[0] for b in bn]
tcip = glob.glob(path + "*TCI*")[0]


with rasterio.open(tcip) as r:
    true_color_image = r.read()
    r, g, b = true_color_image[0], true_color_image[1], true_color_image[2]
    true_color_image = np.dstack((r, g, b)) / 3.5


# they are already in 160m res.
Plot.plot_image(true_color_image)

detector = S2PixelCloudDetector(
    threshold=0.6,
    average_over=4,
    dilation_size=2,
    all_bands=False
)

data = []
prof = None
for p in b_path:
    with rasterio.open(p) as rast:
        prof = rast.profile
        data.append(rast.read().squeeze() / 10000)

data = np.dstack(data)

cloud_prob = detector.get_cloud_probability_maps(data).astype(np.float32)
cloud_mask = detector.get_cloud_masks(data)

Plot.plot_image(mask=cloud_mask)
Plot.plot_image(image=true_color_image, mask=cloud_mask)
Plot.plot_probabilities(true_color_image, cloud_prob)

prof['dtype'] = 'float32'

with rasterio.open(path + "CLD_PROB.tif", "w", **prof) as f:
    f.write(cloud_prob, 1)




