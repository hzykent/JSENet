#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
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
import tensorflow as tf
import numpy as np
import os
from os import makedirs, remove
from os.path import exists, join
import time
import psutil
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Add training ops
        self.add_train_ops(model)

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

        # Name of the snapshot to restore to (None if you want to start from beginning)
        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

    def add_train_ops(self, model):
        """
        Add training ops on top of the model
        """

        ##############
        # Training ops
        ##############

        with tf.variable_scope('optimizer'):

            # Learning rate as a Variable so we can modify it
            self.learning_rate = tf.Variable(model.config.learning_rate, trainable=False, name='learning_rate')

            # Create the gradient descent optimizer with the given learning rate.
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, model.config.momentum)

            # Training step op
            gvs = optimizer.compute_gradients(model.loss)

            if model.config.grad_clip_norm > 0:

                # Get gradient for deformable convolutions and scale them
                scaled_gvs = []
                for grad, var in gvs:
                    if 'offset_conv' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    if 'offset_mlp' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    else:
                        scaled_gvs.append((grad, var))

                # Clipping each gradient independantly
                capped_gvs = [(tf.clip_by_norm(grad, model.config.grad_clip_norm), var) for grad, var in scaled_gvs]

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(capped_gvs)

            else:
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(gvs)

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, model, dataset, debug_NaN=False):
        """
        Train the model on a particular dataset.
        """

        if debug_NaN:
            # Add checking ops
            self.check_op = tf.add_check_numerics_ops()

        # Parameters log file
        if model.config.saving:
            model.parameters_log()

        # Train loop variables
        t0 = time.time()
        self.training_step = 0
        self.training_epoch = 0
        mean_dt = np.zeros(2)
        last_display = t0
        epoch_n = 1
        mean_epoch_n = 0

        # Initialise iterator with train data
        self.sess.run(dataset.train_init_op)

        #--------------------------------------------------------------------------------------------------------------------------------------
        # Training dual network
        if model.config.saving:
            # Training log file
            with open(join(model.saving_path, 'training.txt'), "w") as file:
                file.write('Steps loss_edge_coarse loss_edge_s0 loss_edge_s1 loss_region_coarse loss_region_s0 loss_region_s1 loss_efr_s0 loss_efr_s1 loss_side reg_loss time memory\n')         


            # Killing file (simply delete this file when you want to stop the training)
            if not exists(join(model.saving_path, 'running_PID.txt')):
                with open(join(model.saving_path, 'running_PID.txt'), "w") as file:
                    file.write('Delete this file to stop training')


        # Start loop
        while self.training_epoch < model.config.max_epoch:
            
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = [self.train_op,
                    model.loss_edge_coarse,
                    model.loss_edge_s0,
                    model.loss_edge_s1,
                    model.loss_region_coarse,
                    model.loss_region_s0,
                    model.loss_region_s1,
                    model.loss_efr_s0,
                    model.loss_efr_s1,
                    model.loss_side,
                    model.regularization_loss
                    ]


                # Run normal
                _, L_edge_coarse, L_edge_s0, L_edge_s1, L_region_coarse, L_region_s0, L_region_s1, L_efr_s0, L_efr_s1, L_side, L_reg = self.sess.run(ops, {model.dropout_prob: 0.5})

                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} L_edge_coarse={:5.3f} L_edge_s0={:5.3f} L_edge_s1={:5.3f} L_region_coarse={:5.3f} L_region_s0={:5.3f} L_region_s1={:5.3f} L_efr_s0={:5.3f} L_efr_s1={:5.3f} L_side={:5.3f} L_reg={:5.3f}' \
                            '---{:8.2f} ms/batch (Averaged)'
                    print(message.format(self.training_step,
                                        L_edge_coarse,
                                        L_edge_s0,
                                        L_edge_s1,
                                        L_region_coarse,
                                        L_region_s0,
                                        L_region_s1,
                                        L_efr_s0,
                                        L_efr_s1,
                                        L_side,
                                        L_reg,
                                        1000 * mean_dt[0],
                                        1000 * mean_dt[1]))

                # Log file
                if model.config.saving:
                    process = psutil.Process(os.getpid())
                    with open(join(model.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {:.1f}\n'
                        file.write(message.format(self.training_step,
                                                L_edge_coarse,
                                                L_edge_s0,
                                                L_edge_s1,
                                                L_region_coarse,
                                                L_region_s0,
                                                L_region_s1,
                                                L_efr_s0,
                                                L_efr_s1,
                                                L_side,
                                                L_reg,
                                                t[-1] - t0,
                                                process.memory_info().rss * 1e-6))
                             
                # Check kill signal (running_PID.txt deleted)
                if model.config.saving and not exists(join(model.saving_path, 'running_PID.txt')):
                    break


            except tf.errors.OutOfRangeError:

                # End of train dataset, update average of epoch steps
                mean_epoch_n += (epoch_n - mean_epoch_n) / (self.training_epoch + 1)
                epoch_n = 0
                self.int = int(np.floor(mean_epoch_n))
                model.config.epoch_steps = int(np.floor(mean_epoch_n))
                if model.config.saving:
                    model.parameters_log()

                # Snapshot
                if model.config.saving and (self.training_epoch + 1) % model.config.snapshot_gap == 0:

                    # Tensorflow snapshot
                    snapshot_directory = join(model.saving_path, 'snapshots')
                    if not exists(snapshot_directory):
                        makedirs(snapshot_directory)
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step + 1)

                # Update learning rate
                if self.training_epoch in model.config.lr_decays:
                    op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                            model.config.lr_decays[self.training_epoch]))
                    self.sess.run(op)

                # Increment
                self.training_epoch += 1

                # Reset iterator on training data
                self.sess.run(dataset.train_init_op)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1/0

            # Increment steps
            self.training_step += 1
            epoch_n += 1

        # Remove File for kill signal
        if exists(join(model.saving_path, 'running_PID.txt')):
            remove(join(model.saving_path, 'running_PID.txt'))
        self.sess.close()
