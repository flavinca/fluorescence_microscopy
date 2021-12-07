from nd2reader import ND2Reader
import nd2reader
import skimage
from skimage import data, io, filters
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from scipy import ndimage as ndi
from skimage import (color, feature, filters, measure, morphology, segmentation, util)

threshold = 1000

nd2 = nd2reader.Nd2("./03_A3_40x.nd2")
npfile = np.array(nd2[1])
plt.imshow(npfile)
plt.savefig('image.png')
masknp = npfile

mask = masknp >= threshold
mask2 = masknp < threshold
masknp[mask] = 1
masknp[mask2] = 0
plt.imshow(masknp)
plt.savefig('image2.png')

plt.imshow(masknp)
fig, ax = plt.subplots(figsize=(5, 5))
qcs = ax.contour(masknp, origin='image')
plt.show()
plt.savefig('image3.png')

print(qcs.levels)
print(len(seg) for seg in qcs.allsegs)

thresholds = filters.threshold_multiotsu(masknp, classes=2)
regions = np.digitize(masknp, bins=thresholds)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(masknp)
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].axis('off')
plt.show()
plt.savefig('image4.png')

cells = masknp > thresholds[0]
labeled_cells = measure.label(cells)
im_lab, num_obj = masknp.measure.label(thresholds, return_num=True)
print(num_obj)

plt.imshow(labeled_cells)
plt.show()
plt.savefig('image5.png')

print(labeled_cells)



##Various Attempts
#a = nd2.select(channels="Cy5",z_levels=(0))