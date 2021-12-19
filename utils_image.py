from nd2reader import ND2Reader
import nd2reader
import skimage
from skimage import data, io, filters
import matplotlib.pyplot as plt
import numpy.ma as ma
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage import (color, feature, filters, measure, morphology, segmentation, util)
import csv
import imageio
import pandas as pd


#initialize variables
nd2_file = "./03_A3_40x.nd2"
channel_cells = 1
channel_fluorescence = 0
threshold_val = 1500
intensity_limit = 3000
cellarea_lower = 10
cellarea_upper = 40


#load in files and display image
nd2_image = nd2reader.Nd2("./03_A3_40x.nd2")
npfile_phase = np.array(nd2_image[1])
npfile_fluorescent = np.array(nd2_image[0])
plt.imshow(npfile_phase)
plt.savefig('image1.png')
phase_image = npfile_phase
fluorescent_image = npfile_fluorescent


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
plt.clf()
hist, bins = skimage.exposure.histogram(phase_image)
plt.plot(bins, hist, linewidth=1)
plt.xlim([0,intensity_limit])
plt.xlabel('pixel value (a.u.)')
plt.ylabel('counts')
plt.savefig('image3.png')


#setup threshold and display
threshold_image = phase_image > threshold_val
plt.clf()
plt.set_cmap('Blues')
plt.imshow(threshold_image)
plt.savefig('image4')


#find cells, area. 
np.set_printoptions(threshold=np.inf)
im_lab, num_obj = skimage.measure.label(threshold_image, return_num=True)
cell_pix = np.count_nonzero(im_lab == 3)
cell_area = cell_pix * pixel_microns**2
print("Number of objects found: %s" %num_obj)
print("Pixel count: %s" %cell_pix)
print("Cell Area: %s" %cell_area)


#calculate cell areas
areas = np.zeros(num_obj)
for i in range(num_obj):
    cell = (im_lab == (i+1))
    areas[i] = np.count_nonzero(cell) * pixel_microns**2

    
#determine approved cells based on area
approved_cells = np.zeros_like(threshold_image)
for i in range(num_obj):
    cell = (im_lab == (i+1))
    cell_area = np.count_nonzero(cell) * pixel_microns**2
    if (cell_area > cellarea_lower) & (cell_area < cellarea_upper):
        approved_cells += cell

#show approved cells
approved_lab, num_obj = skimage.measure.label(approved_cells, return_num=True)
print("Approved cells: %s" %num_obj)
plt.clf()
plt.imshow(approved_lab)
plt.savefig('image5')


#load fluorescent image
plt.clf()
plt.set_cmap('Greens')
plt.imshow(fluorescent_image)
plt.savefig('image6')


#find intensity of one cell
cell_id = 5
cell = (approved_lab == cell_id)
cell_fluorescence = cell * fluorescent_image
total_int = np.sum(cell_fluorescence)
print('Cell 5 intensity: %s' % total_int)
plt.clf()
plt.set_cmap('Greens')
plt.imshow(cell_fluorescence)
plt.savefig('image7')


#Measure cell fluorescence
tot_intensities = np.zeros(num_obj) 
mean_intensities = np.zeros(num_obj) 
cell_areas = np.zeros(num_obj)
for i in range(num_obj):
    cell = (approved_lab == i+1)
    cell_areas[i] = np.sum(cell) * pixel_microns**2
    fluorescent_intensity = cell * fluorescent_image
    tot_intensities[i] = np.sum(fluorescent_intensity)
    mean_intensities[i] = np.sum(fluorescent_intensity) / np.count_nonzero(cell)
plt.clf()
plt.hist(tot_intensities, bins=10)
plt.xlabel('total intensity (a.u.)')
plt.ylabel('count')
plt.savefig('image8')
plt.clf()
print(mean_intensities)
    

#generate report of images

df = pd.DataFrame(mean_intensities)
df.to_csv('outputdata.csv')

##Various Attempts
#cells = phase_image > threshold_image[0]
#labeled_cells = measure.label(cells)
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