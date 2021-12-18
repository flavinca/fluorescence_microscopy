from nd2reader import ND2Reader
import nd2reader
import skimage
from skimage import data, io, filters
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from scipy import ndimage as ndi
from skimage import (color, feature, filters, measure, morphology, segmentation, util)


#initialize variables
threshold = 1000
intensity_limit = 3000


#load in file, create mask and display image
nd2 = nd2reader.Nd2("./03_A3_40x.nd2")
npfile = np.array(nd2[1])
plt.imshow(npfile)
plt.savefig('image1.png')
phase_image = npfile


#extract pixel microns from nd2 file
with ND2Reader("./03_A3_40x.nd2") as images:
    pixel_microns = images.metadata['pixel_microns']
print("Pixel micron count: %s" %pixel_microns)


#display phase image
plt.imshow(phase_image)
fig, ax = plt.subplots(figsize=(5, 5))
qcs = ax.contour(phase_image, origin='image')
plt.savefig('image2.png')


#display histogram of intensity 
hist, bins = skimage.exposure.histogram(phase_image)
plt.plot(bins, hist, linewidth=1)
plt.xlim([0,intensity_limit])
plt.xlabel('pixel value (a.u.)')
plt.ylabel('counts')
plt.savefig('image3.png')


#setup threshold and display
threshold_image = phase_image < threshold
plt.clf()
plt.set_cmap('Blues')
plt.imshow(threshold_image)
plt.savefig('image4')


#find cells
cells = phase_image > threshold_image[0]
labeled_cells = measure.label(cells)
im_lab, num_obj = skimage.measure.label(threshold_image, return_num=True)
print("Number of objects found: %s" %num_obj)


bac_1 = im_lab == 2
cell_pix = np.sum(bac_1)
cell_area = cell_pix * pixel_microns**2
print("Pixel count: %s" %cell_pix)
print("Cell Area: %s" %cell_area)


#calculate cell areas
areas = np.zeros(num_obj)
for i in range(num_obj):
    cell = (im_lab == i + 1)
    areas[i] = np.sum(cell) * pixel_microns**2
#print(areas)
    
    


quit()

##Various Attempts
#a = nd2.select(channels="Cy5",z_levels=(0))
#print(qcs.levels)
#print(len(seg) for seg in qcs.allsegs)
#threshold_image = filters.threshold_multiotsu(masknp_image, classes=2)
#regions = np.digitize(masknp_image, bins=threshold_image)
#fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
#ax[0].imshow(mask_image)
#ax[0].set_title('Original')
#ax[0].axis('off')
#ax[1].imshow(regions)
#ax[1].set_title('Multi-Otsu thresholding')
#ax[1].axis('off')
#plt.savefig('image4.png')