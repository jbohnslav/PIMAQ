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

    config = rs.config()
    if options=='default' or options=='brighter':
        resolution_width = 480
        resolution_height = 270
        framerate = 90
    elif options=='large':
        resolution_width = 640
        resolution_height = 480
        framerate=60
    elif options=='calib':
        resolution_width=640
        resolution_height=480
        framerate=6
    else:
        raise NotImplementedError
    config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
    device = Realsense(serial, config, start_t,resolution_height, resolution_width*2, save=save,
                       savedir=datadir, experiment=experiment,
            name=serial_dict[serial],preview=preview,verbose=verbose, options=options,
            movie_format='ffmpeg')

    device.start()
    N_streams =  len(device.prof.get_streams())
    try:
        should_continue = True
        while should_continue:
            # print('acquiring')
            # frames = device.pipeline.wait_for_frames(1000*10)
            # frames =  rs.composite_frame(rs.frame())
            # frames = rs.composite_frame(rs.frame())
            # frames = rs.composite_frame(rs.frame())
            # device.pipeline.poll_for_frames(frames)
            # print(frames.size())
            
            frames = device.pipeline.poll_for_frames()
            # print('frames size: ', frames.size())
            # print('streams size: ', len(streams))
            if frames.size() == N_streams:
                pass
            else:
                time.sleep(1/400)
                continue
            # if frames is not None:
            #     pass
            # else:
            #     time.sleep(1/200)
            #     continue
            # if device.pipeline.poll_for_frame
            # frames = framequeue.wait_for_frame(1000*10)
            # frame = framequeue.poll_for_frame()
            # frames = framequeue.poll_for_frame()
            # print(frames)
            # if frames:
                # print(dir(frames))
            start_t = time.perf_counter()
            # frames = device.pipeline.poll_for_frames()

            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
            frame = device.process(left, right)
            sestime = time.perf_counter() - device.start_t
            cputime = time.time()
            framecount = frames.get_frame_number()
            # by default, milliseconds from 1970. convert to seconds for datetime.datetime
            arrival_time = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)/1000
            timestamp = frames.get_timestamp()/1000
            
            # print('standard process time: %.6f' %(time.perf_counter() - start_t))
            # def write_metadata(self, framecount, timestamp, arrival_time, sestime, cputime):
            metadata = (framecount, timestamp, arrival_time, sestime, cputime)
            if device.save:
                device.save_queue.put_nowait((frame, metadata))
                #             device.write_frame(frame, framecount, timestamp,
                #                 arrival_time, sestime, cputime)
                # if saving, be more stringent about previewing
                # condition = (time.perf_counter()-start_t)*1000<8 and framecount%5==0
              

            # print(time.perf_counter()-start_t)
            if device.preview and framecount % 10 ==0:
                # print(time.perf_counter()-start_t)
                device.preview_queue.put_nowait((frame,framecount))
                if device.latest_frame is not None:
                    cv2.imshow(device.name, device.latest_frame)
                    key = cv2.waitKey(1)
                    if key==27:
                        break
            frames = None
        
    except KeyboardInterrupt:
        print('keyboard interrupt')
        should_continue = False
        # print('User stopped acquisition.')
    finally:
        # don't know why I can't put this in the destructor
        # print(dir(device))
        # if device.preview:
        #     device.preview_queue.put(None)
        #     device.preview_thread.join()
        # time.sleep(1)
        device.stop()

if __name__=='__main__':
    main()