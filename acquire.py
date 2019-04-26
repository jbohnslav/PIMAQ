import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import h5py
import os
import argparse
import multiprocessing as mp
import yaml
import warnings
# import queue
from queue import LifoQueue, Queue, Empty
from threading import Thread
from devices import Realsense

def main():
    if os.path.isfile('serials.yaml'):
        with open('serials.yaml') as f:
            serial_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        warnings.Warn('Need to create a lookup table for matching serial numbers with names.')
        serial_dict = {}
        for serial in serials:
            serial_dict[serial] = None

    serial = '830112071475'
    start_t = time.perf_counter()
    options = 'large'
    save = True
    preview = True
    verbose=True

    datadir = r'D:\DATA\JB\realsense'
    experiment = 'testing_inheritance_%s' %time.strftime('%y%m%d_%H%M%S', time.localtime())
    # name = serial_dict

    
    device = Realsense(serial, start_t,height=None, width=None, save=save,
                       savedir=datadir, experiment=experiment,
            name=serial_dict[serial],preview=preview,verbose=verbose, options=options,
            movie_format='ffmpeg')

    device.start()
    N_streams =  len(device.prof.get_streams())
    device.loop()

if __name__=='__main__':
    main()