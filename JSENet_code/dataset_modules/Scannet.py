#
#
#      0=========================0
#      |    Kernel Point CNN     |
#      0=========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Handle Scannet dataset in a class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Basic libs
import json
import os
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply
from utils.mesh import rasterize_mesh

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

# Dataset parent class
from dataset_modules.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class ScannetDataset(Dataset):
    """
    Class to handle S3DIS dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8, load_test=False):
        Dataset.__init__(self, 'Scannet')

        ###########################
        # Object classes parameters
        ###########################

        # Dict from labels to names
        # We add one entry to the Dict which is the '-1: 'unconsidered''
        self.label_to_names = {
                               -1: 'unconsidered',
                               0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([-1, 0])

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        # cloud_segmentation_region, cloud_segmentation_edge, cloud_segmentation_dual
        self.network_model = 'cloud_segmentation_dual'

        # Number of input threads
        self.num_threads = input_threads

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        self.path = 'Data/Scannet'

        # Path of the training files
        self.train_path = join(self.path, 'training_points')
        self.test_path = join(self.path, 'test_points')

        # Prepare ply files
        self.prepare_ply()

        # List of training and test files
        self.train_files = np.sort([join(self.train_path, f) for f in listdir(self.train_path) if f[-4:] == '.ply'])
        self.test_files = np.sort([join(self.test_path, f) for f in listdir(self.test_path) if f[-4:] == '.ply'])

        # Proportion of validation scenes
        self.validation_clouds = np.loadtxt(join(self.path, 'scannetv2_val.txt'), dtype=np.str)

        # 1 to do validation, 2 to train on all data
        self.validation_split = 1
        self.all_splits = []

        # Load test set or train set?
        self.load_test = load_test


    def prepare_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        paths = [join(self.path, 'scans'), join(self.path, 'scans_test')]
        new_paths = [self.train_path, self.test_path]
        mesh_paths = [join(self.path, 'training_meshes'), join(self.path, 'test_meshes')]

        # Mapping from annot to NYU labels ID
        label_files = join(self.path, 'scannetv2-labels.combined.tsv')
        with open(label_files, 'r') as f:
            lines = f.readlines()
            names1 = [line.split('\t')[1] for line in lines[1:]]
            IDs = [int(line.split('\t')[4]) for line in lines[1:]]
            annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}

        for path, new_path, mesh_path in zip(paths, new_paths, mesh_paths):

            # Create folder
            if not exists(new_path):
                makedirs(new_path)
            if not exists(mesh_path):
                makedirs(mesh_path)

            # Get scene names
            scenes = np.sort([f for f in listdir(path)])
            N = len(scenes)

            for i, scene in enumerate(scenes):

                #############
                # Load meshes
                #############

                # Check if file already done
                if exists(join(new_path, scene + '.ply')):
                    continue
                t1 = time.time()

                # Read mesh
                vertex_data, faces = read_ply(join(path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
                vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

                vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
                if new_path == self.train_path:

                    # Load alignment matrix to realign points
                    align_mat = None
                    with open(join(path, scene, scene + '.txt'), 'r') as txtfile:
                        lines = txtfile.readlines()
                    for line in lines:
                        line = line.split()
                        if line[0] == 'axisAlignment':
                            align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
                    R = align_mat[:3, :3]
                    T = align_mat[:3, 3]
                    vertices = vertices.dot(R.T) + T

                    # Get objects segmentations
                    with open(join(path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                        segmentations = json.load(f)

                    segIndices = np.array(segmentations['segIndices'])

                    # Get objects classes
                    with open(join(path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
                        aggregation = json.load(f)

                    # Loop on object to classify points
                    for segGroup in aggregation['segGroups']:
                        c_name = segGroup['label']
                        if c_name in names1:
                            nyuID = annot_to_nyuID[c_name]
                            if nyuID in self.label_values:
                                for segment in segGroup['segments']:
                                    vertices_labels[segIndices == segment] = nyuID

                ###########################
                # Create finer point clouds
                ###########################

                # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
                points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)

                # Subsample points
                sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)

                # Collect colors from associated vertex
                sub_colors = vertices_colors[sub_vert_inds.ravel(), :]

                if new_path == self.train_path:

                    # Collect labels from associated vertex
                    sub_labels = vertices_labels[sub_vert_inds.ravel()]

                    filename_point = join(new_path, scene + '.ply')
                    
                    # calculate boundaries
                    points_num = sub_points.shape[0]

                    # find semantic boundaries
                    is_boundaries = np.zeros(points_num, dtype=np.int32)

                    # store boundary classes
                    # 0 indicating empty
                    boundary_class_0 = np.zeros(points_num, dtype=np.int32)
                    boundary_class_1 = np.zeros(points_num, dtype=np.int32)
                    boundary_class_2 = np.zeros(points_num, dtype=np.int32)

                    # kd_tree
                    cloud_tree = KDTree(sub_points)
                    
                    # check neighborhood
                    for i in range(points_num):

                        center_class = sub_labels[i]

                        # Pass all unclassified points
                        if center_class == 0:
                            continue
                        
                        # check radius neighbors (r = 0.02)
                        inds = cloud_tree.query_radius(sub_points[i].reshape(1, -1), 0.02)
                        
                        # flag indicating if this point is boundary
                        flag = 0
                        for j in inds[0]:
                            
                            # Pass all unclassified points
                            if sub_labels[j] == 0:
                                continue

                            # when class change is found
                            # 1. boundary of considered class and unconsidered class 
                            # 2. boundary of considered classes
                            if (sub_labels[j] != center_class):
                                # only store considered classes(>0)
                                flag = 1
                                # if center is a point with unconsidered class
                                if center_class == -1:
                                    if boundary_class_0[i] == 0:
                                        boundary_class_0[i] = sub_labels[j]
                                    elif sub_labels[j] == boundary_class_0[i]:
                                        continue
                                    elif boundary_class_1[i] == 0:
                                        boundary_class_1[i] = sub_labels[j]
                                    elif sub_labels[j] == boundary_class_1[i]:
                                        continue
                                    else:
                                        boundary_class_2[i] = sub_labels[j]
                                # if center is a point with considered class
                                else:
                                    boundary_class_0[i] = center_class
                                    if sub_labels[j] == -1:
                                        continue
                                    if boundary_class_1[i] == 0:
                                        boundary_class_1[i] = sub_labels[j]
                                    elif boundary_class_1[i] == sub_labels[j]:
                                        continue
                                    else:
                                        boundary_class_2[i] = sub_labels[j]
                        if flag:
                            is_boundaries[i] = 1


                    # save points
                    write_ply(filename_point,
                                [sub_points, sub_colors, sub_labels, is_boundaries, boundary_class_0, boundary_class_1, boundary_class_2, sub_vert_inds],
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'is_boundary', 'b_c_0', 'b_c_1', 'b_c_2', 'vert_ind'])


                    #############################
                    # Prepare meshes for testing
                    #############################

                    proj_inds = np.squeeze(cloud_tree.query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    vertices_boundaries = is_boundaries[proj_inds]
                    vertices_b_c_0s = boundary_class_0[proj_inds]
                    vertices_b_c_1s = boundary_class_1[proj_inds]
                    vertices_b_c_2s = boundary_class_2[proj_inds]

                    # Save mesh
                    write_ply(join(mesh_path, scene + '_mesh.ply'),
                                    [vertices, vertices_colors, vertices_labels, vertices_boundaries, vertices_b_c_0s, vertices_b_c_1s, vertices_b_c_2s],
                                    ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'is_boundary', 'b_c_0', 'b_c_1', 'b_c_2'],
                                    triangular_faces=faces)


                else:

                    # Save points
                    write_ply(filename_point,
                                [sub_points, sub_colors, sub_vert_inds],
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

                    # Save mesh
                    write_ply(join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'],
                              triangular_faces=faces)


                #  Display
                print('{:s} {:.1f} sec  / {:.1f}%'.format(scene,
                                                          time.time() - t1,
                                                          100 * i / N))

        print('Done in {:.1f}s'.format(time.time() - t0))


    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not exists(tree_path):
            makedirs(tree_path)

        # All training and test files
        files = np.hstack((self.train_files, self.test_files))

        # Initiate containers
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_vert_inds = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_boundaries = {'training': [], 'validation': []}
        self.input_b_c_0 = {'training': [], 'validation': []}
        self.input_b_c_1 = {'training': [], 'validation': []}
        self.input_b_c_2 = {'training': [], 'validation': []}

        # Advanced display
        N = len(files)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(subsampling_parameter))

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]
            if 'train' in cloud_folder:
                if self.validation_split == 1:
                    if cloud_name in self.validation_clouds:
                        self.all_splits += [1]
                        cloud_split = 'validation'
                    else:
                        self.all_splits += [0]
                        cloud_split = 'training'
                else:
                    self.all_splits += [0]
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            if (cloud_split != 'test' and self.load_test) or (cloud_split == 'test' and not self.load_test):
                continue

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_vert_inds = data['vert_ind']
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']
                    sub_boundaries = data['is_boundary']
                    sub_b_c_0 = data['b_c_0']
                    sub_b_c_1 = data['b_c_1']
                    sub_b_c_2 = data['b_c_2']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = data['vert_ind']
                else:
                    int_features = np.vstack((data['vert_ind'], data['class'], data['is_boundary'], data['b_c_0'], data['b_c_1'], data['b_c_2'])).T
                    
                # Subsample cloud
                sub_points, sub_colors, sub_int_features = grid_subsampling(points,
                                                                        features=colors,
                                                                        labels=int_features,
                                                                        sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                if cloud_split == 'test':
                    sub_vert_inds = np.squeeze(sub_int_features)
                    sub_labels = None
                else:
                    sub_vert_inds = sub_int_features[:, 0]
                    sub_labels = sub_int_features[:, 1]
                    sub_boundaries = sub_int_features[:, 2]
                    sub_b_c_0 = sub_int_features[:, 3]
                    sub_b_c_1 = sub_int_features[:, 4]
                    sub_b_c_2 = sub_int_features[:, 5]
                
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                if cloud_split == 'test':
                    write_ply(sub_ply_file,
                                [sub_points, sub_colors, sub_vert_inds],
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                else:
                    write_ply(sub_ply_file,
                                [sub_points, sub_colors, sub_labels, sub_boundaries, sub_b_c_0, sub_b_c_1, sub_b_c_2, sub_vert_inds],
                                ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'is_boundary', 'b_c_0', 'b_c_1', 'b_c_2', 'vert_ind'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_vert_inds[cloud_split] += [sub_vert_inds]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]
                self.input_boundaries[cloud_split] += [sub_boundaries]
                self.input_b_c_0[cloud_split] += [sub_b_c_0]
                self.input_b_c_1[cloud_split] += [sub_b_c_1]
                self.input_b_c_2[cloud_split] += [sub_b_c_2]

            print('', end='\r')
            print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        self.validation_boundaries = []
        self.validation_b_c_0 = []
        self.validation_b_c_1 = []
        self.validation_b_c_2 = []
        self.test_proj = []
        self.test_labels = []
        i_val = 0
        i_test = 0

        # Advanced display
        N = self.num_validation + self.num_test
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Validation projection and labels
            if (not self.load_test) and 'train' in cloud_folder and cloud_name in self.validation_clouds:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels, boundaries, b_c_0, b_c_1, b_c_2 = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'training_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = vertex_data['class']
                    boundaries = vertex_data['is_boundary']
                    b_c_0 = vertex_data['b_c_0']
                    b_c_1 = vertex_data['b_c_1']
                    b_c_2 = vertex_data['b_c_2']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels, boundaries, b_c_0, b_c_1, b_c_2], f)

                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                self.validation_boundaries += [boundaries]
                self.validation_b_c_0 += [b_c_0]
                self.validation_b_c_1 += [b_c_1]
                self.validation_b_c_2 += [b_c_2]
                i_val += 1

            # Test projection
            if self.load_test and 'test' in cloud_folder:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'test_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = np.zeros(vertices.shape[0], dtype=np.int32)

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.test_labels += [labels]
                i_test += 1

            print('', end='\r')
            print(fmt_str.format('#' * (((i_val + i_test) * progress_n) // N), 100 * (i_val + i_test) / N),
                  end='',
                  flush=True)
        
        print('i_val={}'.format(i_val))
        print('i_test={}'.format(i_test))
        print('\n')

        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.epoch_steps * config.batch_num
            random_pick_n = None

        elif split == 'validation':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'test':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        data_split = split

        for i, tree in enumerate(self.input_trees[data_split]):
            # for all the points
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

        ##########################
        # Def generators functions
        ##########################

        def spatially_regular_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pb_list = []
            pbc0_list = []
            pbc1_list = []
            pbc2_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                                                                                  r=config.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / np.square(config.in_radius))
                tukeys[dists > np.square(config.in_radius)] = 0
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                # Safe check for very dense areas
                if n > self.batch_limit:
                    input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                if split in ['test']:
                    input_labels = np.zeros(input_points.shape[0])
                    input_boundaries = np.zeros(input_points.shape[0])
                    input_b_c_0 = np.zeros(input_points.shape[0])
                    input_b_c_1 = np.zeros(input_points.shape[0])
                    input_b_c_2 = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    
                    # converted to zero-based indices
                    input_labels = np.array([self.label_to_idx[l] for l in input_labels])                
                    input_boundaries = self.input_boundaries[data_split][cloud_ind][input_inds]

                    # In Scannet, label is not the same as idx
                    # When represented with label, in b_c_i, 0 indicating unclassified and -1 indicating unconsidered
                    # In idx, 0 indicating unconsidered, 1 indicating unclassified 
                    input_b_c_0 = self.input_b_c_0[data_split][cloud_ind][input_inds]
                    input_b_c_0 = np.array([self.label_to_idx[l] for l in input_b_c_0])

                    input_b_c_1 = self.input_b_c_1[data_split][cloud_ind][input_inds]
                    input_b_c_1 = np.array([self.label_to_idx[l] for l in input_b_c_1])

                    input_b_c_2 = self.input_b_c_2[data_split][cloud_ind][input_inds]
                    input_b_c_2 = np.array([self.label_to_idx[l] for l in input_b_c_2])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.concatenate(pb_list, axis=0),
                           np.concatenate(pbc0_list, axis=0),
                           np.concatenate(pbc1_list, axis=0),
                           np.concatenate(pbc2_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pb_list = []
                    pbc0_list = []
                    pbc1_list = []
                    pbc2_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pb_list += [input_boundaries]
                    pbc0_list += [input_b_c_0]
                    pbc1_list += [input_b_c_1]
                    pbc2_list += [input_b_c_2]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.concatenate(pb_list, axis=0),
                       np.concatenate(pbc0_list, axis=0),
                       np.concatenate(pbc1_list, axis=0),
                       np.concatenate(pbc2_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))

        ###################
        # Choose generators
        ###################

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_gen

        elif split == 'validation':
            gen_func = spatially_regular_gen

        elif split in ['test']:
            gen_func = spatially_regular_gen

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes


    def get_tf_mapping(self, config):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, point_boundaries, point_b_c_0, point_b_c_1, point_b_c_2, stacks_lengths, point_inds, cloud_inds):
            """
            """

            # Get batch indice for each point(indicating the belong cloud)
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]

            # Augmentation : randomly drop colors
            if config.in_features_dim in [4, 5]:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 2:
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
            elif config.in_features_dim == 3:
                stacked_features = stacked_colors
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif config.in_features_dim == 5:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     point_boundaries,
                                                     point_b_c_0,
                                                     point_b_c_1,
                                                     point_b_c_2,
                                                     stacks_lengths,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots]
            input_list += [point_inds, cloud_inds]

            return input_list

        return tf_map


    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Evaluation points are from coarse meshes, not from the ply file we created for our own training
        mesh_path = file_path.split('/')
        mesh_path[-2] = mesh_path[-2][:-6] + 'meshes'
        mesh_path = '/'.join(mesh_path)
        vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
        return np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T



