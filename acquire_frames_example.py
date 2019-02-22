import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import h5py
import os

datadir = r'D:\DATA\JB\realsense\experiment08'
name = 'front_right_JB043'
preview = True
save = True

def append_to_hdf5(f, name, value, axis=0):
    f[name].resize(f[name].shape[axis]+1, axis=axis)
    f[name][-1]=value

def main():
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
    pipeline.start(config)

    if save:
        f = h5py.File(os.path.join(datadir, name+'.h5'), 'w')
        datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
        dset = f.create_dataset('left', (0,), maxshape=(None,),dtype=datatype)
        dset = f.create_dataset('right', (0,), maxshape=(None,),dtype=datatype)
    if preview:
        cv2.namedWindow('Infrared', cv2.WINDOW_AUTOSIZE)
    print('Acquiring...')
    try:
        while True:

            frames = pipeline.wait_for_frames()
            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
            if save:
                ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
                ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
                if ret1 and ret2:
                    append_to_hdf5(f, 'left', left_jpg.squeeze())
                    append_to_hdf5(f, 'right', right_jpg.squeeze())
            if preview:
                cv2.imshow('RealSense', np.vstack((left,right)))
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print('User stopped acquisition.')
    finally:
        if preview:
            cv2.destroyAllWindows()
        if save:
            f.close()
        # Stop streaming
        pipeline.stop()

if __name__=='__main__':
    main()

