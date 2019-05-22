import pyrealsense2 as rs
import numpy as np
import cv2
import time
# import matplotlib.pyplot as plt
import h5py
import os
import argparse
import multiprocessing as mp
import yaml
import warnings
# import queue
from devices import Realsense
import realsense_utils

def initialize_and_loop(config, camname, cam, args, experiment, start_t):

    uncompressed = True if args.options=='calib' else False
    master = True if cam['master'] else False
    serial = 
    if cam['type'] == 'Realsense'
        device = Realsense(cam['serial'], start_t,height=None, width=None, save=args.save,
            savedir=config['savedir'], experiment=experiment,
            name=cam,uncompressed=uncompressed,preview=args.preview,verbose=args.verbose, options=args.options,
            movie_format=args.movie_format)
    elif cam['type'] == 'PointGrey':
        device = PointGrey(serial, start_t, height=512,width=640, save=True, savedir=config['savedir'],
            movie_format=rgs.movie_format, metadata_format='hdf5', uncompressed=False, preview=args.preview,
            verbose=args.verbose, options=cam['options'], name='eye', experiment='testing_pointgrey', 
            strobe=cam['strobe'])
    else:
        raise ValueError('Invalid camera type: %s' %cam['type'])
    # sync_mode = 'master' if serial == args.master else 'slave'
    if serial == args.master:
        device.start()
    else:
        time.sleep(3)
        device.start()
    # runs until keyboard interrupt!
    device.loop()

def main():
    parser = argparse.ArgumentParser(description='Acquire from multiple RealSenses.')
    parser.add_argument('-n','--name', type=str, default='JB999',
        help='Base name for directories. Example: mouse ID')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', 
        help='Configuration for acquisition. Defines number of cameras, serial numbers, etc.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Delete local dirs or not. 0=don''t delete')
    parser.add_argument('-v', '--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')
    parser.add_argument('-o','--options', default='large',
        choices=['default','large', 'calib', 'brighter'], type=str,
        help='Which camera serial number is the "master" camera.')
    parser.add_argument('--movie_format', default='opencv',
        choices=['hdf5','opencv', 'ffmpeg'], type=str,
        help='Method to save files to movies. Dramatically affects performance and filesize')

    args = parser.parse_args()

    if args.movie_format == 'ffmpeg':
        warnings.Warn('ffmpeg uses lots of CPU resources. ' + 
            '60 Hz, 640x480 fills RAM in 5 minutes. Consider opencv')
    serials = realsense_utils.enumerate_connected_devices()
    if args.verbose:
        print('Serials: ', serials)

    if os.path.isfile(args.config):
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        raise ValueError('Invalid config file: %s' %args.config)
    
    """
    Originally I wanted to initialize each device, then pass each device to "run_loop" 
    in its own process. However, pyrealsense2 config objects and pyrealsense2 pipeline objects
    are not pickle-able, and python pickles arguments before passing them to a process. Therefore,
    you have to initialize the configuration and the pipeline from within the process already!
    """
    start_t = time.perf_counter()
    tuples = []
    # make a name for this experiment
    experiment = '%s_%s' %(args.name, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    directory = os.path.join(config['savedir'], experiment)
    if not os.path.isdir(directory):
        os.makedirs(directory)
        with open(os.path.join(directory, 'loaded_config_file.yaml')) as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    for camname, cam in config['cams'].items():
        tup = (config, camname, cam, args, experiment, start_t)
        tuples.append(tup)
    if args.verbose:
        print('Tuples created, starting...')

    with mp.Pool(len(serials)) as p:
        try:
            p.starmap(initialize_and_loop, tuples)  
        except KeyboardInterrupt:
            print('User interrupted acquisition')

    if args.preview:
        cv2.destroyAllWindows()
    
    

if __name__=='__main__':
    main()