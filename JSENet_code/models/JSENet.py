# Basic libs
from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import sys

# Convolution functions
from models.network_blocks import assemble_CNN_blocks, assemble_DCNN_blocks_edge, assemble_DCNN_blocks_region, segmentation_head_edge, segmentation_head_region
from models.network_blocks import side_0_head, side_1_head, side_2_head, side_3_head, side_4_head
from models.network_blocks import segmentation_loss_edge, segmentation_loss_region, bce_loss, segmentation_loss_efr
from models.network_blocks import simple_block, nearest_upsample_block, unary_block, simple_upsample_block, feature_fusion, edge_generation




# ----------------------------------------------------------------------------------------------------------------------
#
#           Model Class
#       \*****************/
#


class JSENet:

    def __init__(self, flat_inputs, config):
        """
        Initiate the model
        :param flat_inputs: List of input tensors (flatten)
        :param config: configuration class
        """

        # Model parameters
        self.config = config

        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path == None:
                self.saving_path = 'results/JSENet_' + self.config.dataset
            else:
                self.saving_path = self.config.saving_path
            if not exists(self.saving_path):
                makedirs(self.saving_path)


        ########
        # Inputs
        ########

        # Sort flatten inputs in a dictionary
        with tf.variable_scope('inputs'):
            self.inputs = dict()
            # flat_inputs[i] corresponding to specific data for a batch of point clouds
            # input point positions for a batch_size of clouds for all network layers
            self.inputs['points'] = flat_inputs[:config.num_layers]
            # corresponding neighbors for all network layers
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_boundaries'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_b_c_0'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_b_c_1'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_b_c_2'] = flat_inputs[ind]
            ind += 1
            self.inputs['convert_neighbors'] = flat_inputs[ind]
            ind += 1
            # zeor based but ignored classes not eliminated
            self.labels = self.inputs['point_labels']
            self.boundaries = self.inputs['point_boundaries']
            self.b_c_0 = self.inputs['point_b_c_0']
            self.b_c_1 = self.inputs['point_b_c_1']
            self.b_c_2 = self.inputs['point_b_c_2']

            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['cloud_inds'] = flat_inputs[ind]

            # Dropout placeholder
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')


        ########
        # Layers
        ########

        # Create layers
        with tf.variable_scope('KernelPointNetwork'):
            extracted_features = assemble_CNN_blocks(self.inputs,
                                                   self.config,
                                                   self.dropout_prob)

            features_edge = assemble_DCNN_blocks_edge(extracted_features,
                                                            self.inputs,
                                                            self.config,
                                                            self.dropout_prob)
            
            features_region = assemble_DCNN_blocks_region(extracted_features,
                                                            self.inputs,
                                                            self.config,
                                                            self.dropout_prob)

            # feature maps of each CNN layer
            feature_64 = extracted_features[0]
            feature_128 = extracted_features[1]
            feature_256 = extracted_features[2]
            feature_512 = extracted_features[3]
            feature_1024 = extracted_features[4]

            training = self.dropout_prob < 0.99


            with tf.variable_scope('region_branch'):

                self.logits_region_coarse = segmentation_head_region(features_region,
                                                            self.config,
                                                            self.dropout_prob)
            

            with tf.variable_scope('edge_branch'):
                # side feature after 1x1 convolution
                side_0_feature_edge = unary_block(-1, -1, feature_64, -1, self.config.num_classes, self.config, training)
                side_1_feature_edge = unary_block(-1, -1, feature_128, -1, self.config.num_classes, self.config, training)
                side_2_feature_edge = unary_block(-1, -1, feature_256, -1, self.config.num_classes, self.config, training)
                side_3_feature_edge = unary_block(-1, -1, feature_512, -1, self.config.num_classes, self.config, training)
                side_4_feature_edge = unary_block(-1, -1, feature_1024, -1, self.config.num_classes, self.config, training)

                # upsample to original size
                r0 = self.config.first_subsampling_dl * self.config.density_parameter
                side_0_feature_edge = simple_block(0, self.inputs, side_0_feature_edge, r0, self.config.num_classes, self.config, training)
                r1 = r0 * 2
                side_1_feature_edge = simple_upsample_block(1, self.inputs, side_1_feature_edge, r1, self.config.num_classes, self.config, training)   
                r2 = r1 * 2
                side_2_feature_edge = nearest_upsample_block(2, self.inputs, side_2_feature_edge, r2, self.config.num_classes, self.config, training)
                side_2_feature_edge = simple_upsample_block(1, self.inputs, side_2_feature_edge, r1, self.config.num_classes, self.config, training)
                r3 = r2 * 2
                side_3_feature_edge = nearest_upsample_block(3, self.inputs, side_3_feature_edge, r3, self.config.num_classes, self.config, training)
                side_3_feature_edge = nearest_upsample_block(2, self.inputs, side_3_feature_edge, r2, self.config.num_classes, self.config, training)
                side_3_feature_edge = simple_upsample_block(1, self.inputs, side_3_feature_edge, r1, self.config.num_classes, self.config, training)
                r4 = r3 * 2
                side_4_feature_edge = nearest_upsample_block(4, self.inputs, side_4_feature_edge, r4, self.config.num_classes, self.config, training)
                side_4_feature_edge = nearest_upsample_block(3, self.inputs, side_4_feature_edge, r3, self.config.num_classes, self.config, training)
                side_4_feature_edge = nearest_upsample_block(2, self.inputs, side_4_feature_edge, r2, self.config.num_classes, self.config, training)
                side_4_feature_edge = simple_upsample_block(1, self.inputs, side_4_feature_edge, r1, self.config.num_classes, self.config, training)
            
                # side_0_logits
                self.side_0_logits_edge = side_0_head(side_0_feature_edge,
                                                self.config,
                                                self.dropout_prob)
                
                # side_1_logits
                self.side_1_logits_edge = side_1_head(side_1_feature_edge,
                                                self.config,
                                                self.dropout_prob)
                
                # side_2_logits
                self.side_2_logits_edge = side_2_head(side_2_feature_edge,
                                                self.config,
                                                self.dropout_prob)
                
                # side_3_logits
                self.side_3_logits_edge = side_3_head(side_3_feature_edge,
                                                self.config,
                                                self.dropout_prob)
                
                # side_4_logits
                self.side_4_logits_edge = side_4_head(side_4_feature_edge,
                                                self.config,
                                                self.dropout_prob)

                # concatenation
                features_edge = tf.concat([features_edge, side_4_feature_edge,
                                                 side_3_feature_edge, side_2_feature_edge, side_1_feature_edge, side_0_feature_edge], axis=1)

                self.logits_edge_coarse = segmentation_head_edge(features_edge,
                                                            self.config,
                                                            self.dropout_prob)


            with tf.variable_scope('refine_module'):
                
                with tf.variable_scope('refinement_region_s0'):
                    
                    # Feature fusion sub-module
                    logits_refine_region_s0 = tf.concat((self.logits_region_coarse, self.logits_edge_coarse), axis=1)                
                    refine_r_feature_s0 = feature_fusion(self.inputs, logits_refine_region_s0, self.config, training)
                    self.logits_region_s0 = segmentation_head_region(refine_r_feature_s0, self.config, self.dropout_prob)

                    # Edge map generation sub-module
                    self.edge_map_fr_s0 = edge_generation(self.inputs, self.logits_region_s0)


                with tf.variable_scope('refinement_edge_s0'):
                    
                    # Edge map generation sub-module
                    self.edge_map_fr_coarse = edge_generation(self.inputs, self.logits_region_coarse)
                    
                    # Feature fusion sub-module
                    logits_edge_coarse_sigmoid = tf.math.sigmoid(self.logits_edge_coarse)
                    logits_refine_edge_s0 = tf.concat((logits_edge_coarse_sigmoid, self.edge_map_fr_coarse), axis=1)
                    refine_e_feature_s0 = feature_fusion(self.inputs, logits_refine_edge_s0, self.config, training)
                    adding_edge_s0 = segmentation_head_edge(refine_e_feature_s0, self.config, self.dropout_prob)
                    self.logits_edge_s0 = self.logits_edge_coarse + adding_edge_s0

                                                    
                with tf.variable_scope('refinement_region_s1'):

                    # Feature fusion sub-module
                    logits_refine_region_s1 = tf.concat((self.logits_region_s0, self.logits_edge_s0), axis=1)
                    refine_r_feature_s1 = feature_fusion(self.inputs, logits_refine_region_s1, self.config, training)
                    self.logits_region_s1 = segmentation_head_region(refine_r_feature_s1, self.config, self.dropout_prob)
                    
                    # Edge map generation sub-module
                    self.edge_map_fr_s1 = edge_generation(self.inputs, self.logits_region_s1)


                with tf.variable_scope('refinement_edge_s1'):
                    
                    # Feature fusion sub-module
                    logits_edge_s0_sigmoid = tf.math.sigmoid(self.logits_edge_s0)
                    logits_refine_edge_s1 = tf.concat((logits_edge_s0_sigmoid, self.edge_map_fr_s0), axis=1)
                    refine_e_feature_s1 = feature_fusion(self.inputs, logits_refine_edge_s1, self.config, training)
                    adding_edge_s1 = segmentation_head_edge(refine_e_feature_s1, self.config, self.dropout_prob)
                    self.logits_edge_s1 = self.logits_edge_s0 + adding_edge_s1


        ########
        # Losses
        ########

        with tf.variable_scope('loss'):
            if len(self.config.ignored_label_inds) > 0:
                # Boolean mask of points that should be ignored
                # 1. unclassified points
                # 2. unconsidered points
                
                # ignored_bool : 1 + 2
                ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
                for ign_label in self.config.ignored_label_inds:
                    ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

                # Collect logits and labels that are not ignored
                inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
                new_logits_region_coarse = tf.gather(self.logits_region_coarse, inds, axis=0)
                new_logits_region_s0 = tf.gather(self.logits_region_s0, inds, axis=0)
                new_logits_region_s1 = tf.gather(self.logits_region_s1, inds, axis=0)
                new_logits_edge_coarse = tf.gather(self.logits_edge_coarse, inds, axis=0)
                new_logits_edge_s0 = tf.gather(self.logits_edge_s0, inds, axis=0)
                new_logits_edge_s1 = tf.gather(self.logits_edge_s1, inds, axis=0)

                new_side_0_logits_edge = tf.gather(self.side_0_logits_edge, inds, axis=0)
                new_side_1_logits_edge = tf.gather(self.side_1_logits_edge, inds, axis=0)
                new_side_2_logits_edge = tf.gather(self.side_2_logits_edge, inds, axis=0)
                new_side_3_logits_edge = tf.gather(self.side_3_logits_edge, inds, axis=0)
                new_side_4_logits_edge = tf.gather(self.side_4_logits_edge, inds, axis=0)

                # new input
                new_dict = {'point_labels': tf.gather(self.labels, inds, axis=0)}
                new_dict['point_boundaries'] = tf.gather(self.boundaries, inds, axis=0)
                new_dict['point_b_c_0'] = tf.gather(self.b_c_0, inds, axis=0)
                new_dict['point_b_c_1'] = tf.gather(self.b_c_1, inds, axis=0)
                new_dict['point_b_c_2'] = tf.gather(self.b_c_2, inds, axis=0)

                # Reduce label values in the range of logit shape
                reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
                inserted_value = tf.zeros((1,), dtype=tf.int32)
                inserted_value = inserted_value - 1
                
                for ign_label in self.config.ignored_label_inds:
                    reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
                
                new_dict['point_labels'] = tf.gather(reducing_list, new_dict['point_labels'])
                # -1 indicates empty in point_b_c_i
                new_dict['point_b_c_0'] = tf.gather(reducing_list, new_dict['point_b_c_0'])
                new_dict['point_b_c_1'] = tf.gather(reducing_list, new_dict['point_b_c_1'])
                new_dict['point_b_c_2'] = tf.gather(reducing_list, new_dict['point_b_c_2'])

                # Add batch weigths to dict if needed
                if self.config.batch_averaged_loss:
                    new_dict['batch_weights'] = self.inputs['batch_weights']

                # Output loss
                    # region branch
                loss_region_coarse = segmentation_loss_region(new_logits_region_coarse,
                                                                    new_dict,
                                                                    batch_average=self.config.batch_averaged_loss)
                loss_region_s0 = segmentation_loss_region(new_logits_region_s0,
                                                      new_dict,
                                                      batch_average=self.config.batch_averaged_loss)
                loss_region_s1 = segmentation_loss_region(new_logits_region_s1,
                                                        new_dict,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_efr_s0 = segmentation_loss_efr(self.edge_map_fr_s0,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss,
                                                        scannet=True,
                                                        inds=inds)
                loss_efr_s1 = segmentation_loss_efr(self.edge_map_fr_s1,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss,
                                                        scannet=True,
                                                        inds=inds)

 
                    # edge branch
                loss_edge_coarse = segmentation_loss_edge(new_logits_edge_coarse,
                                                                new_dict,
                                                                batch_average=self.config.batch_averaged_loss)
                loss_edge_s0 = segmentation_loss_edge(new_logits_edge_s0,
                                                        new_dict,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_edge_s1 = segmentation_loss_edge(new_logits_edge_s1,
                                                        new_dict,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_side_0_edge = bce_loss(new_side_0_logits_edge,
                                                     new_dict,
                                                     batch_average=self.config.batch_averaged_loss)
                loss_side_1_edge = bce_loss(new_side_1_logits_edge,
                                                     new_dict,
                                                     batch_average=self.config.batch_averaged_loss)
                loss_side_2_edge = bce_loss(new_side_2_logits_edge,
                                                     new_dict,
                                                     batch_average=self.config.batch_averaged_loss)
                loss_side_3_edge = segmentation_loss_region(new_side_3_logits_edge,
                                              new_dict,
                                              batch_average=self.config.batch_averaged_loss)
                loss_side_4_edge = segmentation_loss_region(new_side_4_logits_edge,
                                              new_dict,
                                              batch_average=self.config.batch_averaged_loss)


                self.loss_region_coarse = loss_region_coarse * self.config.num_classes
                self.loss_region_s0 = loss_region_s0 * self.config.num_classes
                self.loss_region_s1 = loss_region_s1 * self.config.num_classes
                self.loss_efr_s0 = loss_efr_s0
                self.loss_efr_s1 = loss_efr_s1
                self.loss_edge_coarse = loss_edge_coarse
                self.loss_edge_s0 = loss_edge_s0
                self.loss_edge_s1 = loss_edge_s1
                self.loss_side = loss_side_0_edge + loss_side_1_edge + loss_side_2_edge + loss_side_3_edge + loss_side_4_edge

            else:
                    # region branch
                loss_region_coarse = segmentation_loss_region(self.logits_region_coarse,
                                                                    self.inputs,
                                                                    batch_average=self.config.batch_averaged_loss)
                loss_region_s0 = segmentation_loss_region(self.logits_region_s0,
                                                      self.inputs,
                                                      batch_average=self.config.batch_averaged_loss)
                loss_region_s1 = segmentation_loss_region(self.logits_region_s1,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_efr_s0 = segmentation_loss_efr(self.edge_map_fr_s0,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_efr_s1 = segmentation_loss_efr(self.edge_map_fr_s1,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss)

                    # edge branch
                loss_edge_coarse = segmentation_loss_edge(self.logits_edge_coarse,
                                                                self.inputs,
                                                                batch_average=self.config.batch_averaged_loss)
                loss_edge_s0 = segmentation_loss_edge(self.logits_edge_s0,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_edge_s1 = segmentation_loss_edge(self.logits_edge_s1,
                                                        self.inputs,
                                                        batch_average=self.config.batch_averaged_loss)
                loss_side_0_edge = bce_loss(self.side_0_logits_edge,
                                                     self.inputs,
                                                     batch_average=self.config.batch_averaged_loss)
                loss_side_1_edge = bce_loss(self.side_1_logits_edge,
                                                     self.inputs,
                                                     batch_average=self.config.batch_averaged_loss)
                loss_side_2_edge = bce_loss(self.side_2_logits_edge,
                                                     self.inputs,
                                                     batch_average=self.config.batch_averaged_loss)
                loss_side_3_edge = segmentation_loss_region(self.side_3_logits_edge,
                                              self.inputs,
                                              batch_average=self.config.batch_averaged_loss)
                loss_side_4_edge = segmentation_loss_region(self.side_4_logits_edge,
                                              self.inputs,
                                              batch_average=self.config.batch_averaged_loss)

                self.loss_region_coarse = loss_region_coarse * self.config.num_classes
                self.loss_region_s0 = loss_region_s0 * self.config.num_classes
                self.loss_region_s1 = loss_region_s1 * self.config.num_classes
                self.loss_efr_s0 = loss_efr_s0
                self.loss_efr_s1 = loss_efr_s1
                self.loss_edge_coarse = loss_edge_coarse
                self.loss_edge_s0 = loss_edge_s0
                self.loss_edge_s1 = loss_edge_s1  
                self.loss_side = loss_side_0_edge + loss_side_1_edge + loss_side_2_edge + loss_side_3_edge + loss_side_4_edge

            # Add regularization
            self.loss_coarse = self.loss_region_coarse + self.loss_edge_coarse + self.loss_side
            self.loss_refine =  self.loss_region_s0 + self.loss_region_s1 + self.loss_edge_s0 + self.loss_edge_s1 + self.loss_efr_s0 + self.loss_efr_s1
            self.loss = self.loss_coarse + self.loss_refine + self.regularization_losses()

        return

    def regularization_losses(self):

        #####################
        # Regularization loss
        #####################

        # Get L2 norm of all weights
        regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
        self.regularization_loss = self.config.weights_decay * tf.add_n(regularization_losses)

        ##############################
        # Gaussian regularization loss
        ##############################

        gaussian_losses = []
        for v in tf.global_variables():
            if 'kernel_extents' in v.name:

                # Layer index
                layer = int(v.name.split('/')[1].split('_')[-1])

                # Radius of convolution for this layer
                conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** (layer - 1))

                # Target extent
                target_extent = conv_radius / 1.5
                gaussian_losses += [tf.nn.l2_loss(v - target_extent)]

        if len(gaussian_losses) > 0:
            self.gaussian_loss = self.config.gaussian_decay * tf.add_n(gaussian_losses)
        else:
            self.gaussian_loss = tf.constant(0, dtype=tf.float32)

        return self.gaussian_loss + self.regularization_loss

    def parameters_log(self):

        self.config.save(self.saving_path)


























