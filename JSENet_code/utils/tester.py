# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir
import time
from sklearn.neighbors import KDTree
import pickle

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from sklearn.metrics import precision_recall_curve
from utils.metrics import IoU_from_confusions
from sklearn.metrics import confusion_matrix


from functools import partial


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None, task='SS'):

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')

        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # SS
        # Add a softmax operation for predictions
        self.prob_logits_region = tf.nn.softmax(model.logits_region_s1)

        # SED
        # Add a sigmoid operation for predictions
        self.prob_logits_edge = tf.math.sigmoid(model.logits_edge_s1) + model.edge_map_fr_s1

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_SS(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        self.test_probs = [np.zeros((l.data.shape[0], nc_model), dtype=np.float32) for l in dataset.input_trees['test']]

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions_region')):
                makedirs(join(test_path, 'predictions_region'))
            if not exists(join(test_path, 'probs_region')):
                makedirs(join(test_path, 'probs_region'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()

        while last_min < num_votes:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits_region,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs

                # Average timing
                t += [time.time()]
                #print(batches.shape, stacked_probs.shape, 1000*(t[1] - t[0]), 1000*(t[2] - t[1]))
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['test'])))

                i0 += 1

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['test'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))
                print([np.mean(pots) for pots in dataset.potentials['test']])

                if last_min + 10 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):

                        # Reproject probs
                        probs = self.test_probs[i_test][dataset.test_proj[i_test], :]

                        # Insert false columns for ignored labels
                        probs2 = probs.copy()
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.int32)

                        # Save ascii preds
                        cloud_name = file_path.split('/')[-1]
                        ascii_name = join(test_path, 'predictions_region', cloud_name[:-4] + '.txt')
                        np.savetxt(ascii_name, preds, fmt='%d')
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(dataset.test_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return


    def test_SS_on_val(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        self.test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]

        # Number of points per class in validation set
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.validation_labels])
                i += 1

        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_predictions_region')):
                makedirs(join(test_path, 'val_predictions_region'))
            if not exists(join(test_path, 'val_probs_region')):
                makedirs(join(test_path, 'val_probs_region'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()

        while last_min < num_votes:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits_region,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['validation'])))

                i0 += 1


            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['validation'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i_test in range(dataset.num_validation):

                        # Insert false columns for ignored labels
                        probs = self.test_probs[i_test]
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = dataset.input_labels['validation'][i_test]

                        # Confs
                        Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                        if label_value in dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                    if (epoch_ind + 1) % 10 == 0:

                        # Project predictions
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                        t1 = time.time()
                        files = dataset.train_files
                        i_val = 0
                        proj_probs = []
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:
                                print(i)
                                # Reproject probs on the evaluations points
                                probs = self.test_probs[i_val][dataset.validation_proj[i_val], :]
                                proj_probs += [probs]
                                i_val += 1

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Show vote results
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i_test in range(dataset.num_validation):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    proj_probs[i_test] = np.insert(proj_probs[i_test], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                            # Confusion
                            targets = dataset.validation_labels[i_test]
                            Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                            if label_value in dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                        # Save predictions
                        print('Saving clouds')
                        t1 = time.time()
                        files = dataset.train_files
                        i_test = 0
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:

                                # Get points
                                points = dataset.load_evaluation_points(file_path)

                                # Get the predicted labels
                                preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                                # Project potentials on original points
                                pots = dataset.potentials['validation'][i_test][dataset.validation_proj[i_test]]

                                # Save plys
                                cloud_name = file_path.split('/')[-1]
                                test_name = join(test_path, 'val_predictions_region', cloud_name)
                                write_ply(test_name,
                                          [points, preds, pots, dataset.validation_labels[i_test]],
                                          ['x', 'y', 'z', 'preds', 'pots', 'gt'])
                                test_name2 = join(test_path, 'val_probs_region', cloud_name)
                                prob_names = ['_'.join(dataset.label_to_names[label].split())
                                              for label in dataset.label_values]
                                write_ply(test_name2,
                                          [points, proj_probs[i_test]],
                                          ['x', 'y', 'z'] + prob_names)
                                i_test += 1
                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(dataset.val_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return


    def test_SED_on_val(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over test clouds
        nc_model = model.config.num_classes
        test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]


        # Test saving path
        if model.config.saving:
            test_path = join('test', model.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_probs_edge')):
                makedirs(join(test_path, 'val_probs_edge'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        i0 = 0
        epoch_ind = 0
        mean_dt = np.zeros(2)
        last_display = time.time()
        while epoch_ind < num_votes:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits_edge,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions per cloud
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices (ignored points are not eliminated)
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    test_probs[c_i][inds] = test_smooth * test_probs[c_i][inds] + (1-test_smooth) * probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['validation'])))

                i0 += 1


            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['validation'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))

                if (epoch_ind + 1) % 30 == 0:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.train_files
                    i_val = 0
                    proj_probs = []
                    for i, file_path in enumerate(files):
                        if dataset.all_splits[i] == dataset.validation_split:

                            # Reproject probs on the evaluations points
                            probs = test_probs[i_val][dataset.validation_proj[i_val], :]
                            proj_probs += [probs]
                            i_val += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    print('MF on full clouds')
                    t1 = time.time()
                    class_ind = 0               # 0 ~ num_class - 1: indicating channel number of outputs
                    class_MFs = []

                    for l_ind, label_value in enumerate(dataset.label_values):
                        if label_value not in dataset.ignored_labels:
                            
                            proj_probs_class = []

                            for proj_p in proj_probs:
                                proj_probs_class += [proj_p[:, class_ind]]
                            
                            preds_class = []
                            target_class = []
                            # bci: original label values and ignored classes not eliminated
                                # 0 indicating empty and unclassified in Scannet
                                # -1 indicating empty in S3DIS
                            for i_test in range(dataset.num_validation):
                                # prediction for class_i
                                pre_class = proj_probs_class[i_test]

                                # boundaries_class: indicating if this point is boundary for class i
                                boundaries_class_0 = [(l == label_value) for l in dataset.validation_b_c_0[i_test]]
                                boundaries_class_1 = [(l == label_value) for l in dataset.validation_b_c_1[i_test]]
                                boundaries_class_2 = [(l == label_value) for l in dataset.validation_b_c_2[i_test]]
                                boundaries_class = [(l0 or l1 or l2) for l0, l1, l2 in zip(boundaries_class_0, boundaries_class_1, boundaries_class_2)]
                                
                                if len(dataset.ignored_labels) > 0:
                                    # Boolean mask of points that should be ignored
                                    ignored_bool = np.zeros_like(dataset.validation_labels[i_test], dtype=np.bool)
                                    for ign_val in dataset.ignored_labels:
                                        ignored_bool = np.logical_or(ignored_bool, np.equal(dataset.validation_labels[i_test], ign_val))
                                    
                                    # inds that are not ignored
                                    inds = np.squeeze(np.where(np.logical_not(ignored_bool)))

                                    # select points that are not ignored
                                    pre_class = np.array(pre_class)[inds]
                                    boundaries_class = np.array(boundaries_class)[inds]

                                preds_class = np.hstack((preds_class, pre_class))
                                target_class = np.hstack((target_class, boundaries_class))
                            
                            if np.sum(target_class) == 0:
                                raise ValueError('This class does not exist in the testing set')
                            precisions, recalls, thresholds = precision_recall_curve(target_class, preds_class)
                            f1_scores = []
                            for (precision, recall) in zip(precisions, recalls):
                                if recall + precision == 0:
                                    f1_scores += [0.0]
                                else:
                                    f1_scores += [2*recall*precision/(recall+precision)]
                            # class_MF: maximal F meature for class i
                            class_MF = np.max(f1_scores)
                            print('class_{}:{}'.format(class_ind, class_MF))

                            class_MFs += [class_MF]

                            class_ind = class_ind + 1
                    
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))


                    mMF = np.mean(class_MFs)
                    s = '{:5.2f} | '.format(100 * mMF)
                    for MF in class_MFs:
                        s += '{:5.2f} '.format(100 * MF)
                    print('-' * len(s))
                    print(s)
                    print('-' * len(s) + '\n')
                    with open(join(test_path, 'MFs.txt'), "a") as file:
                        file.write('MF on full clouds\n')
                        file.write('mean MF = {:.3f}\n'.format(mMF))
                        file.write('-' * len(s))
                        file.write(s)
                        file.write('-' * len(s) + '\n')

                    # Save predictions
                    print('Saving clouds')
                    t1 = time.time()
                    files = dataset.train_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        if dataset.all_splits[i] == dataset.validation_split:

                            # Get points
                            points = dataset.load_evaluation_points(file_path)

                            # Save plys
                            cloud_name = file_path.split('/')[-1]
                            test_name = join(test_path, 'val_probs_edge', cloud_name)
                            prob_names = ['_'.join(dataset.label_to_names[label].split()) for label in dataset.label_values
                                      if label not in dataset.ignored_labels]
                            write_ply(test_name,
                                      [points, proj_probs[i_test], dataset.validation_b_c_0[i_test], dataset.validation_b_c_1[i_test], dataset.validation_b_c_2[i_test]],
                                      ['x', 'y', 'z'] + prob_names + ['b_c_0', 'b_c_1', 'b_c_2'])
                            i_test += 1
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                
                self.sess.run(dataset.val_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return
