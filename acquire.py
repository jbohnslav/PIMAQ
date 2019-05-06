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

datadir = r'D:\DATA\JB\realsense'

def initialize_and_loop(serial,args,datadir, experiment, name,start_t):

    uncompressed = True if args.options=='calib' else False

    
    device = Realsense(serial, start_t,height=None, width=None, save=args.save,
        savedir=datadir, experiment=experiment,
        name=name,uncompressed=uncompressed,preview=args.preview,verbose=args.verbose, options=args.options,
        movie_format=args.movie_format)

    # sync_mode = 'master' if serial == args.master else 'slave'
    if serial == args.master:
        device.start(sync_mode='master')
    else:
        time.sleep(3)
        device.start()
    # runs until keyboard interrupt!
    device.loop()

def main():
    parser = argparse.ArgumentParser(description='Acquire from multiple RealSenses.')
    parser.add_argument('-m','--mouse', type=str, default='JB999',
        help='ID of mouse for file naming.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Delete local dirs or not. 0=don''t delete')
    parser.add_argument('-v', '--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')
    parser.add_argument('--master', default='830112071475',type=str,
        help='Which camera serial number is the "master" camera.')
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
    assert(args.master in serials)
    if os.path.isfile('serials.yaml'):
        with open('serials.yaml') as f:
            serial_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        warnings.Warn('Need to create a lookup table for matching serial numbers with names.')
        serial_dict = {}
        for serial in serials:
            serial_dict[serial] = None
    """
    Originally I wanted to initialize each device, then pass each device to "run_loop" 
    in its own process. However, pyrealsense2 config objects and pyrealsense2 pipeline objects
    are not pickle-able, and python pickles arguments before passing them to a process. Therefore,
    you have to initialize the configuration and the pipeline from within the process already!
    """
    start_t = time.perf_counter()
    tuples = []
    # make a name for this experiment
    experiment = '%s_%s' %(args.mouse, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    for serial in serials:

        tup = (serial, args, datadir, experiment, serial_dict[serial],start_t)
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