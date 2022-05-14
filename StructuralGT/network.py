#Draft
import numpy as np
import igraph as ig
import os
import cv2 as cv
from StructuralGT import error, base, process_image#, convert
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
import time
import functools
import gsd.hoomd
from skimage.morphology import skeletonize_3d, disk, ball
import pandas as pd
import copy

import StructuralGT

#import convert

class _crop():
    """
    Crop object
    dim is dimensions: 2 or 3
    dims is lengths: eg (500,1500,10)
    crop is crop point locations eg (200,300,200,300)
    """
    
    def __init__(self, Network, domain=None):
        self.dim = Network.dim
        
        ## Find depth and surface of network and assign to _crop.depth; _crop.surface ##
        olist = np.asarray(sorted(os.listdir(Network.stack_dir)))
        mask = list(base.Q_img(olist[i]) for i in range(len(olist)))
        fname = sorted(olist[mask])[0] #First name
        self.surface = int(os.path.splitext(fname)[0][5:]) #Strip file type and 'slice' then convert to int
        if Network._2d:
            self.depth = 1
        else:
            if domain == None:
                self.depth = sum(mask)
            else:
                self.depth = domain[5]-domain[4]
            if self.depth == 0:
                raise error.ImageDirectoryError(Network.stack_dir)
        
        ## Assign dims and crop ##
        if domain is None:
            self.crop = slice(None)
            planar_dims = cv.imread(Network.stack_dir+'/slice'+str(0)+'.tiff',cv.IMREAD_GRAYSCALE).shape
            if self.dim == 2:
                self.dims = (1,) + planar_dims
            else:
                self.dims = (self.depth,) + planar_dims 
                
        else:
            self.crop = slice(domain[2],domain[3]),slice(domain[0],domain[1])
            if self.dim == 2: 
                self.dims = (1,domain[3]-domain[2],domain[1]-domain[0])
            else:
                #self.crop = slice(domain[0],domain[1]),slice(domain[2],domain[3]),slice(domain[4],domain[5])
                self.dims = (domain[5]-domain[4],domain[3]-domain[2],domain[1]-domain[0])
                
    def rotationals():
        if self.dim == 2:
            self.centre = 0.5*np.array([self.crop[0]+self.crop[1],self.crop[2]+self.crop[3]])


class Network():
    """
    Generic SGT graph class: a specialised case of the igraph Graph object with 
    additional attributes defining geometric features, associated images,
    dimensionality etc.
    
    Initialised from directory containing raw image data
    self._2d determined from the number of images with identical dimensions (suggesting a stack when > 1)
    
    Image shrinking/cropping is carried out at the gsd stage in analysis.
    I.e. full images are binarized but cropping their graphs may come after
    """
    def __init__(self, directory, child_dir='/Binarized'):
        if not isinstance(directory, str):
            raise TypeError
            
        self.dir = directory
        self.child_dir = child_dir
        self.stack_dir = self.dir + self.child_dir
        self.rotate = None
        self.Q = None
        self.crop = None

        shape = []
        self.img = []
        for fname in sorted(os.listdir(self.dir)):
            if base.Q_img(fname):
                _slice = cv.imread(self.dir+'/'+fname,cv.IMREAD_GRAYSCALE)
                self.img.append(_slice)
                shape.append(_slice.shape)

        if len(set(shape)) == len(shape):
            self._2d = True
            self.dim = 2
        else:
            self._2d = False
            self.dim = 3
        
    def binarize(self, options_dict=None):
        """Binarizes stack of experimental images using a set of image processing parameters in options_dict.
        Note this enforces that all images have the same shape as the first image encountered by the for loop.
        (i.e. the first alphanumeric titled image file)
        """
        
        if options_dict is None:
            options = self.dir + '/img_options.json'
            with open(options) as f:
                options_dict = json.load(f)

        if not os.path.isdir(self.dir + self.child_dir):
            os.mkdir(self.dir + self.child_dir)

        i=0
        for name in sorted(os.listdir(self.dir)):
            if not base.Q_img(name):
                continue
            else:
                img_exp = cv.imread(self.dir+'/'+name,cv.IMREAD_GRAYSCALE)
                if i == 0: shape = img_exp.shape
                elif img_exp.shape != shape: continue
                _, img_bin, _ = process_image.binarize(img_exp, options_dict)
                plt.imsave(self.dir+self.child_dir+'/slice'+str(i)+'.tiff', img_bin, cmap=cm.gray)
                i+=1

        self.options = options_dict

    def stack_to_gsd(self, name='skel.gsd', crop=None, skeleton=True, rotate=None, debubble=None):
        """Writes a .gsd file from the object's directory.
        The name of the written .gsd is set as an attribute so it may be easily matched with its Graph object 
        Running this also sets the positions, shape attributes
        """
        if crop is None and rotate is not None: raise ValueError('If rotating a graph, crop must be specified')
        start = time.time()
        self.type = 'rectangular'
        if name[0] == '/':
            self.gsd_name = name
        else:
            self.gsd_name = self.stack_dir + '/' + name
        self.gsd_dir = os.path.split(self.gsd_name)[0]       
        
        if rotate is not None:
            #Calculate outer crop
            #(i.e. that which could contain any rotation of the inner crop)
            #Use it to write the unrotated skel.gsd
            centre = (crop[0]+0.5*(crop[1]-crop[0]),crop[2]+0.5*(crop[3]-crop[2]))
            diagonal = ((crop[1]-crop[0])**2+(crop[3]-crop[2])**2)**0.5

            outer_crop = np.array([centre[0]-diagonal*0.5,
                                   centre[0]+diagonal*0.5,
                                   centre[1]-diagonal*0.5,
                                   centre[1]+diagonal*0.5], dtype=int)
            inner_crop = crop
            self.inner_crop = inner_crop
            crop = outer_crop
        cropper = _crop(self, domain=crop)
        
        #Initilise i such that it starts at the lowest number belonging to the images in the stack_dir
        #First require boolean mask to filter out non image files
        img_bin = np.zeros(cropper.dims)
        i = cropper.surface
        for fname in sorted(os.listdir(self.stack_dir)):
            if base.Q_img(fname) and i<cropper.depth:
                img_bin[i] = cv.imread(self.stack_dir+'/slice'+str(i)+'.tiff',cv.IMREAD_GRAYSCALE)[cropper.crop]/255
                i=i+1
            else:
                continue

        #For 2D images, img_bin_3d.shape[0] == 1
        self.img_bin_3d = img_bin
        self.img_bin = img_bin

        assert self.img_bin_3d.shape[1] > 1
        assert self.img_bin_3d.shape[2] > 1
        
        self.img_bin_3d = self.img_bin            #Always 3d, even for 2d images
        self.img_bin = np.squeeze(self.img_bin)   #3d for 3d images, 2d otherwise

        if self._2d:
            assert self.img_bin_3d.shape[1] == self.img_bin.shape[0]
            assert self.img_bin_3d.shape[2] == self.img_bin.shape[1]
        else:
            self.img_bin_3d.shape == self.img_bin.shape
        
        if skeleton:
            self.skeleton = skeletonize_3d(np.asarray(self.img_bin, dtype=int))
            self.skeleton_3d = skeletonize_3d(np.asarray(self.img_bin_3d, dtype=int))
            #self.skeleton_3d = np.swapaxes(self.skeleton_3d, 1, 2)
        else:
            self.img_bin = np.asarray(self.img_bin)

        positions = np.asarray(np.where(np.asarray(self.skeleton_3d) == 1)).T
        self.shape = np.asarray(list(max(positions.T[i])+1 for i in (0,1,2)[0:self.dim]))
        self.positions = positions
        self.img = self.img[0]

        with gsd.hoomd.open(name=self.gsd_name, mode='wb') as f:
            s = gsd.hoomd.Snapshot()
            s.particles.N = len(positions)
            s.particles.position = base.shift(positions)
            s.particles.types = ['A']
            s.particles.typeid = ['0']*s.particles.N
            f.append(s)

        end = time.time()
        print('Ran stack_to_gsd() in ', end-start, 'for gsd with ', len(positions), 'particles')

        if debubble is not None: self = base.debubble(self, debubble)

        assert self.img_bin.shape == self.skeleton.shape
        assert self.img_bin_3d.shape == self.skeleton_3d.shape

        """Set rot matrix attribute for later"""
        if rotate is not None:
            from scipy.spatial.transform import Rotation as R
            r = R.from_rotvec(rotate/180*np.pi * np.array([0, 0, 1]))
            self.rotate = r.as_matrix()
            self.crop = np.asarray(outer_crop) - min(outer_crop)
            
        self.cropper = cropper
        
           
    def stack_to_circular_gsd(self, radius, name='circle.gsd', rotate=None, debubble=None, skeleton=True):
        """Writes a cicular .gsd file from the object's directory.
        Currently only capable of 2D graphs
        Unlike stack_to_gsd, the axis of rotation is not the centre of the image, but the point (radius,radius)
        The name of the written .gsd is set as an attribute so it may be easily matched with its Graph object 
        Running this also sets the positions, shape attributes.
        
        Note the rotation implementation is very different to self.stack_to_gsd():
        A rotating circular graph will never lose/gain nodes so no need to recalculate weights
        Instead
            Generate the graph at theta=0.
            Set all attributes.
            Apply rotation matrix to positional attributes
            
        TODO: Change positions ... != 0 to == 1
        """
        start = time.time()
        self.type = 'circle'
        if name[0] == '/':
            self.gsd_name = name
        else:
            self.gsd_name = self.stack_dir + '/' + name
        self.gsd_dir = os.path.split(self.gsd_name)[0]
        img_bin=[]
        
        #Initilise i such that it starts at the lowest number belonging to the images in the stack_dir
        #First require boolean mask to filter out non image files
        olist = np.asarray(sorted(os.listdir(self.stack_dir)))
        mask = list(base.Q_img(olist[i]) for i in range(len(olist)))
        if len(mask) == 0:
            raise error.ImageDirectoryError(self.stack_dir)
        fname = sorted(olist[mask])[0] #First name
        i = int(os.path.splitext(fname)[0][5:]) #Strip file type and 'slice' then convert to int
        
        #img_slice = cv.imread(self.stack_dir+'/slice'+str(i)+'.tiff',cv.IMREAD_GRAYSCALE)
        
        #Read the image
        for fname in sorted(os.listdir(self.stack_dir)):
            if base.Q_img(fname):
                img_slice = cv.imread(self.stack_dir+'/slice'+str(i)+'.tiff',cv.IMREAD_GRAYSCALE)/255
                img_bin.append(img_slice)
                i=i+1
            else:
                continue
                
        #For 2D images, img_bin_3d.shape[0] == 1
        img_bin = np.asarray(img_bin)
        
        self.img_bin_3d = img_bin            #Always 3d, even for 2d images
        self.img_bin = np.squeeze(img_bin)   #3d for 3d images, 2d otherwise
        
        assert self._2d

        canvas = np.ones(self.img_bin.shape)
        disk_pos = np.asarray(np.where(disk(radius)!=0)).T
        canvas[disk_pos[0], disk_pos[1]] = 0
        self.img_bin = np.ma.MaskedArray(self.img_bin, mask=canvas)
        self.img_bin = np.ma.filled(self.img_bin, fill_value=0)
        
        canvas = np.ones(self.img_bin_3d.shape)
        disk_pos = np.asarray(np.where(disk(radius)!=0)).T
        disk_pos = np.array([np.zeros(len(disk_pos)), disk_pos.T[0], disk_pos.T[1]], dtype=int)
        canvas[disk_pos[0], disk_pos[1], disk_pos[2]] = 0
        self.img_bin_3d = np.ma.MaskedArray(self.img_bin_3d, mask=canvas)
        self.img_bin_3d = np.ma.filled(self.img_bin_3d, fill_value=0)
        self.img_bin = self.img_bin_3d[0]
        
        assert self.img_bin_3d.shape[1] > 1
        assert self.img_bin_3d.shape[2] > 1
        
        if skeleton:
            self.skeleton = skeletonize_3d(np.asarray(self.img_bin))
            self.skeleton_3d = skeletonize_3d(np.asarray(self.img_bin_3d))
        else:
            self.img_bin = np.asarray(self.img_bin)
        
        positions = np.asarray(np.where(np.asarray(self.skeleton_3d) != 0)).T
        self.shape = np.asarray(list(max(positions.T[i])+1 for i in (0,1,2)[0:self.dim]))
        self.positions = positions
        
        with gsd.hoomd.open(name=self.gsd_name, mode='wb') as f:
            s = gsd.hoomd.Snapshot()
            s.particles.N = len(positions)
            s.particles.position = base.shift(positions)
            s.particles.types = ['A']
            s.particles.typeid = ['0']*s.particles.N
            f.append(s)
        
        end = time.time()
        print('Ran stack_to_gsd() in ', end-start, 'for gsd with ', len(positions), 'particles')
        
        if debubble is not None: self = base.debubble(self, debubble)
        
        assert self.img_bin.shape == self.skeleton.shape
        assert self.img_bin_3d.shape == self.skeleton_3d.shape    
        
        """Set rot matrix attribute for later"""
        if rotate is not None:
            from scipy.spatial.transform import Rotation as R
            r = R.from_rotvec(rotate/180*np.pi * np.array([0, 0, 1]))
            self.rotate = r.as_matrix()
            
        
    def G_u(self, **kwargs):
        """
        Sets igraph object as an attribute
        """
        if 'sub' not in kwargs: kwargs['sub'] = True
        G =  base.gsd_to_G(self.gsd_name, _2d = self._2d, sub=kwargs['sub'])
        
        self.Gr = G
        #self.Gr_copy = copy.deepcopy(G)
        
        if len(kwargs)!=0:
            if 'sub' in kwargs: kwargs.pop('sub')
            if 'weight_type' in kwargs: self.Gr = base.add_weights(self, **kwargs)

        self.shape = list(max(list(self.Gr.vs[i]['o'][j] for i in range(self.Gr.vcount()))) for j in (0,1,2)[0:self.dim])

        if self.rotate is not None:
            centre = np.asarray(self.shape)/2
            inner_length = (self.inner_crop[1]-self.inner_crop[0])*0.5
            inner_crop = np.array([centre[0]-inner_length,
                                   centre[0]+inner_length,
                                   centre[1]-inner_length,
                                   centre[1]+inner_length], dtype=int)
            
            node_positions = np.asarray(list(self.Gr.vs[i]['o'] for i in range(self.Gr.vcount())))
            node_positions = base.oshift(node_positions, _shift=centre)
            node_positions = np.vstack((node_positions.T, np.zeros(len(node_positions)))).T
            node_positions = np.matmul(node_positions, self.rotate).T[0:2].T
            node_positions = base.shift(node_positions, _shift=-centre)
            drop_list = []
            for i in range(self.Gr.vcount()):
                if not base.Q_inside(np.asarray([node_positions[i]]), inner_crop):
                    drop_list.append(i)
                    continue
                    
                self.Gr.vs[i]['o'] = node_positions[i]
                self.Gr.vs[i]['pts'] = node_positions[i]
            self.Gr.delete_vertices(drop_list)

            node_positions = np.asarray(list(self.Gr.vs[i]['o'] for i in range(self.Gr.vcount())))
            final_shift = np.min(list(node_positions.T[i] for i in range(self.dim)))
            edge_positions_list = np.asarray(list(base.oshift(self.Gr.es[i]['pts'], _shift=centre) for i in range(self.Gr.ecount())))
            for i, edge in enumerate(edge_positions_list):
                edge_position = np.vstack((edge.T, np.zeros(len(edge)))).T
                edge_position = np.matmul(edge_position, self.rotate).T[0:2].T
                edge_position = base.shift(edge_position, _shift=-centre+final_shift)
                self.Gr.es[i]['pts'] = edge_position
            
            
            node_positions = base.shift(node_positions, _shift=final_shift)
            for i in range(self.Gr.vcount()): 
                self.Gr.vs[i]['o'] = node_positions[i]
                self.Gr.vs[i]['pts'] = node_positions[i]
                
            self.shape = list(max(list(self.Gr.vs[i]['o'][j] for i in range(self.Gr.vcount()))) for j in (0,1,2)[0:self.dim])

    def weighted_Laplacian(self, weights='weight'):

        L=np.asarray(self.Gr.laplacian(weights=weights))
        self.L = L

    #Labelling function which takes a graph object, node attribute and writes their values to a new .gsd file. 
    def Node_labelling(self, attribute, attribute_name, filename, edge_weight=None):
        """
        Method saves a new .gsd which has the graph in self.Gr labelled with the node attributes in attribute
        Method saves all the main attributes of a Network object in the .gsd such that the network object may be loaded from the file
        """

        assert self.Gr.vcount() == len(attribute)

        #save_name = os.path.split(prefix)[0] + '/'+attribute_name + os.path.split(prefix)[1]
        if filename[0] == '/':
            save_name = filename
        else:
            save_name = self.stack_dir + '/' + filename
        if os.path.exists(save_name):
            mode = 'rb+'
        else:
            mode = 'wb'

        f = gsd.hoomd.open(name=save_name, mode=mode)
        self.labelled_name = save_name
        
        #Must segregate position list into a node_position section and edge_position
        node_positions = np.asarray(list(self.Gr.vs()[i]['o'] for i in range(self.Gr.vcount())))
        positions = node_positions
        for edge in self.Gr.es():
            positions=np.vstack((positions,edge['pts']))
        positions = np.unique(positions, axis=0)
        if self._2d:
            node_positions = np.hstack((np.zeros((len(node_positions),1)),node_positions))
            positions = np.hstack((np.zeros((len(positions),1)),positions))

        #node_positions = base.shift(node_positions)
        #positions = base.shift(positions)
        s = gsd.hoomd.Snapshot()
        N = len(positions)
        s.particles.N = N
        s.particles.position = positions
        s.particles.types = ['Edge', 'Node']
        s.particles.typeid = [0]*N
        L = list(max(positions.T[i])*2 for i in (0,1,2))
        #s.configuration.box = [L[0]/2, L[1]/2, L[2]/2, 0, 0, 0]
        s.configuration.box = [1,1,1,0,0,0]
        s.log['particles/'+attribute_name] = [np.NaN]*N
        
        #To store graph, must first convert sparse adjacency matrix as 3 dense matrices
        #rows, columns, values = convert.to_dense(np.array(self.Gr.get_adjacency(attribute=edge_weight).data,dtype=np.single))
        #s.log['Adj_rows'] = rows
        #s.log['Adj_cols'] = columns
        #s.log['Adj_values'] = values
        #s.log['img_options'] = self.options
        
        #Store optional Network attributes
        #if self.Q is not None: s.log['InvLaplacian'] = self.Q

        start = time.time()

        j=0
        for i,particle in enumerate(positions):
            node_id = np.where(np.all(positions[i] == node_positions, axis=1) == True)[0]
            if len(node_id) == 0: 
                continue
            else:
                s.log['particles/'+attribute_name][i] = attribute[node_id[0]]
                s.particles.typeid[i] = 1
                j+=1

        f.append(s)
     
    def recon(self, axis, surface, depth):

        #Method displays 2D slice of binary image and annotates with attributes from 3D graph subslice

        Gr_copy = copy.deepcopy(self.Gr)

        #self.Gr = base.sub_G(self.Gr)

        axis_0 = abs(axis-2)

        display_img = np.swapaxes(self.img_bin_3d, 0, axis_0)[surface]
        drop_list=[]
        for i in range(self.Gr.vcount()):
            if self.Gr.vs[i]['o'][axis_0] < surface or self.Gr.vs[i]['o'][axis_0] > surface+depth:
                drop_list.append(i)
                continue

        self.Gr.delete_vertices(drop_list)

        node_positions = np.asarray(list(self.Gr.vs()[i]['o'] for i in range(self.Gr.vcount())))
        positions = np.array([[0,0,0]])
        for edge in self.Gr.es():
            positions=np.vstack((positions,edge['pts']))


        fig = plt.figure(figsize=(10,25))
        plt.scatter(node_positions.T[2], node_positions.T[1], s=10, color='red')
        plt.scatter(positions.T[2], positions.T[1], s=2)
        plt.imshow(self.img_bin[axis], cmap=cm.gray)
        plt.show()

        self.Gr = Gr_copy

    
class ResistiveNetwork(Network):
    """Child of generic SGT Network class.
    Equipped with methods for analysing resistive flow networks
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def potential_distribution(self, plane, boundary1, boundary2, R_j=0, rho_dim=1):
        """
        Solves for the potential distribution in a weighted network.
        Source and sink nodes are connected according to a penetration boundary condition.
        Sets the corresponding weighted Laplacian, potential and flow attributes.
        The 'plane' arguement defines the axis along which the boundary arguements refer to.
        R_j='infinity' enables the unusual case of all edges having the same unit resistance.
        
        NOTE: Critical that self.G_u() is called before every self.potential_distribution() call
        TODO: Remove this requirement or add an error to warn
        """
        #self.G_u(weight_type=['Conductance'], R_j=R_j, rho_dim=rho_dim) #Assign weighted graph attribute
        #self.Gr = base.sub_G(self.Gr)

        if R_j != 'infinity':
            weight_array = np.asarray(self.Gr.es['Conductance']).astype(float)
            weight_array = weight_array[~np.isnan(weight_array)]
            self.edge_weights = weight_array
            weight_avg =np.mean(weight_array)
        else:
            self.Gr_connected.es['Conductance'] = np.ones(self.Gr.ecount())
            weight_avg = 1

        #Add source and sink nodes:
        self.Gr_connected = self.Gr
        source_id = max(self.Gr_connected.vs).index + 1
        sink_id = source_id + 1
        self.Gr_connected.add_vertices(2)

        print('Graph has max ', self.shape)
        axes = np.array([0,1,2])[0:self.dim]
        indices = axes[axes!=plane]
        plane_centre1 = np.zeros(self.dim, dtype=int)
        delta = np.zeros(self.dim, dtype=int)
        delta[plane] = 10 #Arbitrary. Standardize?
        for i in indices: plane_centre1[i] = self.shape[i]/2
        plane_centre2 = np.copy(plane_centre1)
        plane_centre2[plane] = self.shape[plane]
        source_coord = plane_centre1 - delta 
        sink_coord = plane_centre2 + delta
        print('source coord is ', source_coord)
        print('sink coord is ', sink_coord)
        self.Gr_connected.vs[source_id]['o'] = source_coord
        self.Gr_connected.vs[sink_id]['o'] = sink_coord

    #Connect nodes on a given boundary to the external current nodes
        print('Before connecting external nodes, G has vcount ', self.Gr_connected.vcount())
        for node in self.Gr_connected.vs:
            if node['o'][plane] > boundary1[0] and node['o'][plane] < boundary1[1]:
                self.Gr_connected.add_edges([(node.index, source_id)])
                self.Gr_connected.es[self.Gr_connected.get_eid(node.index,source_id)]['Conductance'] = weight_avg
                self.Gr_connected.es[self.Gr_connected.get_eid(node.index,source_id)]['pts'] = base.connector(source_coord,node['o'])
            if node['o'][plane] > boundary2[0] and node['o'][plane] < boundary2[1]:
                self.Gr_connected.add_edges([(node.index, sink_id)])
                self.Gr_connected.es[self.Gr_connected.get_eid(node.index,sink_id)]['Conductance'] = weight_avg 
                self.Gr_connected.es[self.Gr_connected.get_eid(node.index,sink_id)]['pts'] = base.connector(sink_coord,node['o'])

    #Write skeleton connected to external node
        print(self.Gr_connected.is_connected(), ' connected')
        print('After connecting external nodes, G has vcount ', self.Gr_connected.vcount())
        connected_name = os.path.split(self.gsd_name)[0] + '/connected_' + os.path.split(self.gsd_name)[1] 
        #connected_name = self.stack_dir + '/connected_' + self.gsd_name 
        base.G_to_gsd(self.Gr_connected, connected_name)
        
        if R_j=='infinity': self.L = np.asarray(self.Gr.laplacian())
        else: self.weighted_Laplacian(weights='Conductance')
        
        F = np.zeros(sink_id+1)
        print(self.L.shape, 'L')
        F[source_id] = 1
        F[sink_id] = -1

        Q = np.linalg.pinv(self.L, hermitian=True)
        P = np.matmul(Q,F)

        self.P = P
        self.F = F
        self.Q = Q
        
    def effective_resistance(self, source=-1, sink=-2):

        O_eff = self.Q[source,source]+self.Q[sink,sink]-2*self.Q[source,sink]
        
        return O_eff
    

class StructuralNetwork(Network):
    """
    Child of generic SGT Network class.
    Equipped with methods for analysing structural networks
    """
    def __init__(self, directory):
        super().__init__(directory)
        
    def G_calc(self):
        avg_indices = dict()

        operations = [self.Gr.diameter, self.Gr.density, self.Gr.transitivity_undirected, self.Gr.assortativity_degree]
        names = ['Diameter', 'Density', 'Clustering', 'Assortativity by degree']

        for operation,name in zip(operations,names):
            start = time.time()
            avg_indices[name] = operation()
            end = time.time()
            print('Calculated ', name, ' in ', end-start)
            
        self.G_attributes = avg_indices
        
    def node_calc(self, Betweenness=True, Closeness=True, Degree=True):
        if Betweenness: self.Betweenness = self.Gr.betweenness()
        if Closeness: self.Closeness = self.Gr.closeness()
        if Degree: self.Degree = self.Gr.degree()
