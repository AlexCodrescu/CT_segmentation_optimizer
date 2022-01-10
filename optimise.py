import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import skimage
from skimage import filters
from skimage import feature
from skimage import measure
from skimage import morphology, segmentation,io
import cv2
from scipy.ndimage import distance_transform_edt
from skimage import color

import csv

import argparse
import sys
import os

from PIL import Image

import time
start_time = time.time()

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser()
    parser.add_argument('inputDirectory',
                    help='Path to the input HU file.')
    parser.add_argument('inputDirectory',
                    help='Path to the input seg file.')
    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.inputDirectory):
       print("Files exist")

with open(sys.argv[2], 'r') as f:
    seg = [[int(num) for num in line.split(' ')] for line in f]

with open(sys.argv[1], 'r') as f:
    hu = [[int(num) for num in line.split(' ')] for line in f]


#returns median haunsfield value for the pixels in the original segmentation.
def med_val(seg,hu):
    med=[]
    for i in range(0,512) :
        for j in range(0,512) :
            if seg[i][j] == 1 :
                med.append(hu[i][j])
    return(median(med))

#returns denoise image of hu
def denoise(hu, show=True):
    hu_denoise = filters.median(hu, selem=np.ones((5,5)))
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5))
    if show==True :
        ax1.imshow(hu_denoise)
        plt.show()
    return hu_denoise

#creates edges from hu
def create_edges(hu,sig = 3, show = True):
    pixels = hu
    array = np.array(pixels, dtype=np.uint8)
    new_image = Image.fromarray(array)
    image = np.asanyarray(new_image)

    edges = skimage.feature.canny(image, sigma = sig)
    if show == True:
        plt.imshow(edges)
        plt.show()
    return edges

#isolates values from hu based on the median value of original segmentation
def hu_isolate(hu,seg,low = 20, high= 20, show=True):
    hu_isolated=denoise(hu,show=False)
    med=med_val(seg,hu)
    for b in range(0,512):
        for a in range(0,512):
            if not (med-low) <= hu[b][a] <= (med+high) :
                hu_isolated[b][a]=0
    if show == True:
        plt.imshow(hu_isolated)
        plt.show()
    return hu_isolated

#returns the center of the segmentation
def center(seg):
    pixels = seg
    array = np.array(pixels, dtype=np.uint8)
    new_image = Image.fromarray(array)
    image = np.asanyarray(new_image)

    # compute the center of the contour
    M = cv2.moments(image)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX,cY

#returns number of contours
def nr_cnts(seg):
    pixels = seg
    array = np.array(pixels, dtype=np.uint8)
    new_image = Image.fromarray(array)
    image = np.asanyarray(new_image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

#separates each original segmentation from seg file in separated matrixes
def separate_sgmts(seg, show = False):
    pixels = seg
    array = np.array(pixels, dtype=np.uint8)
    new_image = Image.fromarray(array)
    image = np.asanyarray(new_image)
    cnt, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    A=[[]]
    for i in range(len(cnt)):
        output_canvas = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(output_canvas, cnt, i, (255,255,255), -1)
        if show == True :
            plt.imshow(output_canvas)
            plt.show()
        A.append(output_canvas)
    return A

#creates mask based on the original segmentation
def create_mask (hu,seg):
    hu_isolated=hu_isolate(hu,seg,low=15,high=15,show=False)
    edges=create_edges(hu_isolated,sig=3,show=False)
    middle=center(seg)

    dt=distance_transform_edt(~edges)
    #plt.imshow(dt)
    #plt.show()

    #local=feature.peak_local_max(dt,indices=True,min_distance=25)
    #plt.plot(local[:,1],local[:,0],'r.')
    #plt.imshow(dt)
    #plt.show()

    local_max=feature.peak_local_max(dt,indices=False,min_distance=25)
    markers=measure.label(local_max)

    labels=morphology.watershed(-dt,markers)
    #plt.imshow(segmentation.mark_boundaries(image,labels))
    #plt.imshow((segmentation.mark_boundaries(image,labels)*255).astype(np.uint8))
    #plt.show()

    final=np.zeros(shape=(512,512), dtype = "uint8")
    #plt.imshow(labels)
    #plt.show()

    # loop over the unique segment values
    for (i, segVal) in enumerate(np.unique(labels)):

        # construct a mask for the segment
        print ("[x] inspecting segment %d" % (i))
        mask = np.zeros(shape=(512,512), dtype = "uint8")
        mask[labels == segVal] = 1

        #show the masked region
        #cv2.imshow("mask",mask)
        #plt.imshow(mask)
        #plt.show()
        #cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))

        if mask[middle[1]][middle[0]]==1:
                final = mask

        cv2.waitKey(0)

    return final

#splits each segmentation from seg file into separated segmentations then creates masks for each then combines them into one
def opt(hu,seg):
    cnt=nr_cnts(seg)

    if cnt == 1:
        return create_mask(hu,seg)

    if cnt > 1:
        separated_seg = separate_sgmts(seg)
        final = np.zeros(shape=(512, 512), dtype="uint8")
        for i in range (1,cnt + 1):
            A=list(separated_seg[i][:])
            A=transf(A)
            B=create_mask(hu,A)
            for j in range (1,512):
                for k in range (1,512):
                    if B[j][k] > 0:
                        final[j][k]=1
        return final

#transforms 255 pixels in 1
def transf(b):
    for i in range(0,512):
        for j in range(0,512):
            if b[i][j]==255:
                b[i][j]=1
    return b

#plt.imshow(hu)
#plt.colorbar()
#plt.show()

#plt.imshow(seg)
#plt.show()

optim=opt(hu,seg)
np.savetxt('optim.out',optim,fmt='%i')

print("--- %s seconds ---" % (time.time() - start_time))