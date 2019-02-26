import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import h5py
import os
import argparse

datadir = r'D:\DATA\JB\realsense\experiment08'
name = 'front_right_JB043'
preview = True
save = True

def append_to_hdf5(f, name, value, axis=0):
    f[name].resize(f[name].shape[axis]+1, axis=axis)
    f[name][-1]=value

def main():
    parser = argparse.ArgumentParser(description='Acquire from single RealSense.')
    parser.add_argument('-n','--name', type=str, default=name,
        help='Name of experiment for file naming.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Delete local dirs or not. 0=don''t delete')
    parser.add_argument('--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')

    args = parser.parse_args()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    resolution_width = 480
    resolution_height = 270
    framerate = 90
    config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()
    ir_sensors = device.query_sensors()[0] # 1 for RGB
    # get options example
    # exp = ir_sensors.get_option(rs.option.exposure)
    # lims = ir_sensors.get_option_range(rs.option.exposure)
    # print('Exposure: {0}'.format(exp))
    # print('Min: {0} Max: {1} Default: {2} Step: {3}'.format(lims.min, lims.max, 
    #                                                    lims.default, lims.step))
    ir_sensors.set_option(rs.option.enable_auto_exposure,0)
    ir_sensors.set_option(rs.option.exposure,500)
    ir_sensors.set_option(rs.option.gain,16)

    if args.save:
        f = h5py.File(os.path.join(datadir, args.name+'.h5'), 'w')
        datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
        dset = f.create_dataset('left', (0,), maxshape=(None,),dtype=datatype)
        dset = f.create_dataset('right', (0,), maxshape=(None,),dtype=datatype)
        dset = f.create_dataset('sestime',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('cputime',(0,),maxshape=(None,),dtype=np.float64)
    if args.preview:
        cv2.namedWindow('Infrared', cv2.WINDOW_AUTOSIZE)
    print('Acquiring...')
    start_t = time.perf_counter()
    try:
        while True:

            frames = pipeline.wait_for_frames()
            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
            time_acq = time.perf_counter() - start_t
            cputime = time.time()
            if args.save:
                ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
                ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
                if ret1 and ret2:
                    append_to_hdf5(f, 'left', left_jpg.squeeze())
                    append_to_hdf5(f, 'right', right_jpg.squeeze())
                    append_to_hdf5(f,'sestime', time_acq)
                    append_to_hdf5(f, 'cputime', cputime)
            if args.preview:
                cv2.imshow('Infrared', np.vstack((left,right)))
                key = cv2.waitKey(1)
                if key==27:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print('User stopped acquisition.')
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        if args.save:
            f.close()
        # Stop streaming
        pipeline.stop()

if __name__=='__main__':
    main()

