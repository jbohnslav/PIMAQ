try:
    import pyrealsense2 as rs
except ImportError as e:
    print('pyrealsense not found, cannot acquire realsense cameras...')
    rs = None
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
if rs is not None:
    from devices import Realsense 
try:
    from devices import PointGrey
except ImportError as e:
    print('PySpin not found, can''t acquire from FLIR cameras')
    PointGrey = None
import realsense_utils

def initialize_and_loop(config, camname, cam, args, experiment, start_t):

    if cam['type'] == 'Realsense':
        device = Realsense(serial=cam['serial'], 
            start_t=start_t,
            options=config['realsense_options'],
            save=args.save,
            savedir=config['savedir'], 
            experiment=experiment,
            name=camname,
            movie_format=args.movie_format,
            metadata_format='hdf5', 
            uncompressed=config['realsense_options']['uncompressed'],
            preview=args.preview,
            verbose=args.verbose,
            master=cam['master'],
            codec=config['codec']
            )
    elif cam['type'] == 'PointGrey':
        device = PointGrey(serial=cam['serial'], 
            start_t=start_t, 
            options=cam['options'],
            save=args.save, 
            savedir=config['savedir'],
            experiment=experiment,
            name=camname,
            movie_format=args.movie_format, 
            metadata_format='hdf5', 
            uncompressed=False, # setting to False always because you don't need to calibrate it
            preview=args.preview,
            verbose=args.verbose,
            strobe=cam['strobe'],
            codec=config['codec']
            )
    else:
        raise ValueError('Invalid camera type: %s' %cam['type'])
    # sync_mode = 'master' if serial == args.master else 'slave'
    if cam['master']:
        sleep_time = np.random.randn()+3
        time.sleep(sleep_time)
        device.start()
    else:
        sleep_time = np.random.randn()+3
        time.sleep(sleep_time)
        device.start()
    # runs until keyboard interrupt!
    device.loop()

def main():
    parser = argparse.ArgumentParser(description='Multi-camera acquisition in Python.')
    parser.add_argument('-n','--name', type=str, default='JB999',
        help='Base name for directories. Example: mouse ID')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', 
        help='Configuration for acquisition. Defines number of cameras, serial numbers, etc.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Use this flag to save to disk. If not passed, will only view')
    parser.add_argument('-v', '--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')
    parser.add_argument('--movie_format', default='opencv',
        choices=['hdf5','opencv', 'ffmpeg', 'directory'], type=str,
        help='Method to save files to movies. Dramatically affects performance and filesize')

    args = parser.parse_args()

    if args.movie_format == 'ffmpeg':
        warnings.Warn('ffmpeg uses lots of CPU resources. ' + 
            '60 Hz, 640x480 fills RAM in 5 minutes. Consider opencv')
    if rs is not None:
        serials = realsense_utils.enumerate_connected_devices()
        if args.verbose:
            print('Realsense Serials: ', serials)

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
    if args.save:
        directory = os.path.join(config['savedir'], experiment)
        if not os.path.isdir(directory):
            os.makedirs(directory)
            with open(os.path.join(directory, 'loaded_config_file.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    for camname, cam in config['cams'].items():
        tup = (config, camname, cam, args, experiment, start_t)
        tuples.append(tup)
    if args.verbose:
        print('Tuples created, starting...')

    if len(config['cams']) >1 :
        with mp.Pool(len(config['cams'])) as p:
            try:
                p.starmap(initialize_and_loop, tuples)  
            except KeyboardInterrupt:
                p.close()
                p.join()
                print('User interrupted acquisition')
    else:
        assert len(tuples) == 1
        initialize_and_loop(*tuples[0])

    if args.preview:
        cv2.destroyAllWindows()
    
    

if __name__=='__main__':
    main()