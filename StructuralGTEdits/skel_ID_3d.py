"""skel_ID: A collection of methods and tools for analyzing
and altering a skeletal image.  Prepares the skeleton for
conversion into a graph object.

Copyright (C) 2021, The Regents of the University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contributers: Drew Vecchio, Samuel Mahler, Mark D. Hammig, Nicholas A. Kotov
Contact email: vecdrew@umich.edu
"""

from __main__ import *

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize_3d
from skimage.morphology import disk, remove_small_objects
from skimage.morphology import binary_dilation as dilate

#from StructuralGT import skel_features as sk
#from StructuralGT import base
import skel_features as sk
import base

import cv2 as cv
import os
import shutil
#import gsd.hoomd

# Return a sparse 3D array with 1s at the locations of branch points. Branch point features defined by 3x3x3 array
# and generated by skel_features.py
def branchedPoints(skel):
    # Defin rotationally independent features
    x_base = np.array([[0,0,0],
                     [-1,1,0],
                     [-1,-1,0],
                      [1,1,0],
                      [1,-1,0]])

    _3d_x_base = np.array([[0,0,0],
                     [-1,1,0],
                     [-1,-1,0],
                      [1,1,0],
                      [1,-1,0],
                      [0,0,1],
                      [0,0,-1]])

    quadrapod_base = np.array([[0,0,0],
                             [1,1,1],
                             [1,-1,1],
                             [-1,1,1],
                           [-1,-1,1]])

    tripod_base = np.array([[0,0,0],
                             [1,1,1],
                             [1,-1,1],
                             [-1,1,1]])

    quad_pyramid_base = np.array([[0,0,0],
                             [1,1,1],
                             [1,-1,1],
                             [-1,1,1],
                             [-1,-1,1]])

    tri_pyramid_base = np.array([[0,0,0],
                             [1,1,1],
                             [1,-1,1],
                             [-1,1,1]])

    t_base = np.array([[0,0,0],
                      [-1,0,0],
                      [1,0,0],
                      [0,1,0]])

    y_base = np.array([[0,0,0],
                     [-1,1,0],
                     [1,1,0],
                     [0,-1,0]])

    names = ("cross","3d_cross", "quadrapod", "tripod", "quad_pyramid", "tri_pyramid", "tee","y")
    bases = (x_base, _3d_x_base, quadrapod_base, tripod_base, quad_pyramid_base, tri_pyramid_base, t_base, y_base)

    for name,base  in zip(names,bases):
        sk.write_feature(base, name, "branch_lib")

    br = 0
    for FeatName in os.listdir("branch_lib"):
        ArrList = np.load("branch_lib/" + FeatName)
        for Array in ArrList:
            br = br + ndimage.binary_hit_or_miss(skel, Array)

    return np.asarray(np.where(br != 0))

def endPoints(skel):

    end_base = np.array([[0,0,0],
                        [ 0,0,1]])

    sk.write_feature(end_base, "end", "end_lib")

    ep = 0
    ArrList = np.load("end_lib/end.npy")
    for Array in ArrList:
        ep = ep + ndimage.binary_hit_or_miss(skel, Array)

    return np.asarray(np.where(ep != 0))

def pruning(skeleton, size, Bps):
    branchpoints = Bps
    #remove iteratively end points "size" times from the skeleton
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        points = np.logical_and(endpoints, branchpoints)
        endpoints = np.logical_xor(endpoints, points)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton


def merge_nodes(skeleton):

    # overlay a disk over each branch point and find the overlaps to combine nodes
    skeleton_integer = 1 * skeleton
    radius = 2
    mask_elem = disk(radius)
    BpSk = branchedPoints(skeleton_integer)
    BpSk = 1*(dilate(BpSk, mask_elem))

    # widenodes is initially an empty image the same size as the skeleton image
    sh = skeleton_integer.shape
    widenodes = np.zeros(sh, dtype='int')

    # this overlays the two skeletons
    # skeleton_integer is the full map, BpSk is just the branch points blown up to a larger size
    for x in range(sh[0]):
        for y in range(sh[1]):
            if skeleton_integer[x, y] == 0 and BpSk[x, y] == 0:
                widenodes[x, y] = 0
            else:
                widenodes[x, y] = 1

    # reskeletonzing widenodes and returning it, nearby nodes in radius 2 of each other should have been merged
    newskel = skeletonize(widenodes)
    return newskel

# Instead of returning skeletons and feature points coordinates, this returns skeletons but writes the features (and skeleton) to gsd
# Note that 'coord' suffix indicates the variable is a list of coordinates. Otherwise, data representing skeletons/features is a sparse array
def make_skel(params, merge, prune, clean, r_size, aspect=(1,1,1)):
    
    #Change from master: img_bin is now a stack of 2D .tiffs
    img_bin = []
    i=0
    for name in sorted(os.listdir(params['directory']+'/Binarized')):
        if base.Q_img(name):
            img_slice = cv.imread(params['directory']+'/Binarized/'+str(name),cv.IMREAD_GRAYSCALE)
            if img_slice is not None:
                img_bin.append(img_slice)
                i=i+1
            else:
                pass
        else:
            pass
    print(img_bin)
    skeleton = skeletonize_3d(np.asarray(img_bin)/255).astype(int)

    # calling the three functions for merging nodes, pruning edges, and removing disconnected segments
    if(merge == 1):
        skeleton = merge_nodes(skeleton)

    if(clean == 1):
        skeleton = remove_small_objects(skeleton, r_size, connectivity=3)

    if(prune == 1):
        skeleton = pruning(skeleton, 500, Bps)



    branchs = branchedPoints(skeleton)
    ends = endPoints(skeleton)
    skels = np.asarray(np.where(skeleton!=0)) #Outputs array with shape (3,N)
    

    for feature in (skels, branchs, ends):
        for i in (0,1,2):
            feature[i] = feature[i]*aspect[i]
    
    skel_coords = skels.T
    branch_coords = branchs.T
    end_coords = ends.T
    Q_gsd = False
    if Q_gsd:
        s = gsd.hoomd.snapshot()
        with gsd.hoomd.open(name=params['save_name']+'_raw.gsd', mode='wb') as f:
            s.particles.N = np.shape(skel_coords)[0] + np.shape(branch_coords)[0] + np.shape(end_coords)[0]
            s.particles.position = skel_coords
            f.append(s)

#    with gsd.hoomd.open(name=params['save_name']+'_cleaned.gsd', mode='wb') as f:
#        pass



   #clean_skel = skeleton
    if Q_gsd==False:
        return skeleton, skel_coords, branch_coords, end_coords
