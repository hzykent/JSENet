# Common libs
import time
import os
import numpy as np
import argparse

# My libs
from utils.config import Config
from utils.tester import ModelTester

# models
from models.JSENet import JSENet

# Datasets
from dataset_modules.S3DIS import S3DISDataset
from dataset_modules.Scannet import ScannetDataset


# select testing semantic segmentation task or semantic edge detection task
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='SS',help="SS/SED")
FLAGS = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def test_caller(path, step_ind, on_val):

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    config.validation_size = 500


    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    if config.dataset == 'S3DIS':
        dataset = S3DISDataset(config.input_threads)
        on_val = True
    elif config.dataset == 'Scannet':
        dataset = ScannetDataset(config.input_threads, load_test=(not on_val))
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Create subsample clouds of the models
    dl0 = config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)

    # Initialize input pipelines
    if on_val:
        dataset.init_input_pipeline(config)
    else:
        dataset.init_test_input_pipeline(config)


    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    model = JSENet(dataset.flat_inputs, config)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    if FLAGS.task == 'SS':
        tester = ModelTester(model, restore_snap=chosen_snap, task='SS')
    else:
        tester = ModelTester(model, restore_snap=chosen_snap, task='SED')
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')

    if FLAGS.task == 'SS':
        if config.dataset.startswith('S3DIS'):
            tester.test_SS_on_val(model, dataset)
        elif config.dataset.startswith('Scannet'):
            if on_val:
                tester.test_SS_on_val(model, dataset)
            else:
                tester.test_SS(model, dataset)
        else:
            raise ValueError('Unsupported dataset')
    else:
        if config.dataset.startswith('S3DIS'):
            tester.test_SED_on_val(model, dataset)
        elif config.dataset.startswith('Scannet'):
            if on_val:
                tester.test_SED_on_val(model, dataset)
            else:
                raise ValueError('SED task can only be tested on validation set since gt data of Scannet test set is not available')
        else:
            raise ValueError('Unsupported dataset')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Choose the model to test
    ##########################

    chosen_log = 'results/S3DIS_pre'

    #
    #   You can also choose the index of the snapshot to load (last by default)
    #

    chosen_snapshot = -1

    #
    #   Eventually, you can choose to test your model on the validation set
    #

    on_val = True

    #
    #   If you want to modify certain parameters in the Config class, for example, to stop augmenting the input data,
    #   there is a section for it in the function "test_caller" defined above.
    #

    ###########################
    # Call the test initializer
    ###########################

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    test_caller(chosen_log, chosen_snapshot, on_val)



