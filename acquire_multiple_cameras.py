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

datadir = r'D:\DATA\JB\realsense\experiment08'
name = 'front_right_JB043'
preview = True
save = False

def append_to_hdf5(f, name, value, axis=0):
    f[name].resize(f[name].shape[axis]+1, axis=axis)
    f[name][-1]=value

def enumerate_connected_devices(context):
    """
    Enumerate the connected Intel RealSense devices
    Parameters:
    -----------
    context           : rs.context()
                         The context created for using the realsense library
    Return:
    -----------
    connect_device : array
                       Array of enumerated devices which are connected to the PC
    """
    connect_device = []
    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))
    return connect_device
class Device:
    def __init__(self, config, serial,savedir=None,experiment=None, name=None,
        save_format='hdf5',preview=False):

        self.config = config
        self.serial = serial
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile

        self.name = name
        self.preview = preview
        if fileobj:
            self.save = True
            self.fileobj = fileobj
        else:
            self.save = False

    def start(self):
        pipeline = rs.pipeline()
        self.config.enable_device(self.serial)
        try:
            pipeline_profile = pipeline.start(config)
        except RuntimeError:
            print('Pipeline for camera %s already running, restarting...' %serial)
            pipeline.stop()
            time.sleep(1)
            pipeline.start(config)
        self.pipeline = pipeline
        self.prof = pipeline_profile
        self.update_settings()
        if self.save:
            self.initialize_saving()
        if self.preview:
            self.initialize_preview()

    def initialize_preview(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def initialize_saving(self):
        if self.save_format == 'hdf5':
            f = h5py.File(os.path.join(self.savedir, fname), 'w')
            datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
            dset = f.create_dataset('left', (0,), maxshape=(None,),dtype=datatype)
            dset = f.create_dataset('right', (0,), maxshape=(None,),dtype=datatype)
            dset = f.create_dataset('sestime',(0,),maxshape=(None,),dtype=np.float64)
            dset = f.create_dataset('cputime',(0,),maxshape=(None,),dtype=np.float64)
            self.fileobj = f
        else:
            raise NotImplementedError

    def update_settings(self):
        sensor = self.prof.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled,1)
        
        device = self.prof.get_device()
        ir_sensors = device.query_sensors()[0] # 1 for RGB
        ir_sensors.set_option(rs.option.enable_auto_exposure,0)
        ir_sensors.set_option(rs.option.exposure,500)
        ir_sensors.set_option(rs.option.gain,16)

    def write_frames(self,left,right, time_acq,cputime):
        if not hasattr(self, 'fileobj'):
            raise ValueError('Writing for camera %s not initialized.' %self.camname)
        ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
        ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
        if ret1 and ret2:
            append_to_hdf5(f, 'left', left_jpg.squeeze())
            append_to_hdf5(f, 'right', right_jpg.squeeze())
            append_to_hdf5(f,'sestime', time_acq)
            append_to_hdf5(f, 'cputime', cputime)

    def stop_streaming(self):
        self.pipeline.stop()

    def __del__(self): 
        self.stop_streaming()
        if self.save:
            self.fileobj.close()
        if self.preview:
            cv2.destroyWindow(self.name)
        print('Destructor called, cam %s deleted.' %self.name) 

def run_loop(device):
    start_t = time.perf_counter()
    framecount = 0
    if device.preview:
        font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            frames = device.pipeline.wait_for_frames()
            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
            time_acq = time.perf_counter() - start_t
            cputime = time.time()
            if device.save:
                device.write_frames(left, right, time_acq, cputime)

            if device.preview:
                out = np.vstack((left,right))
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                        
                # string = '%.4f' %(time_acq*1000)
                string = '%s:%07d' %(device.name)
                cv2.putText(out,string,(10,500), font, 0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow(serial, out)
                key = cv2.waitKey(1)
                if key==27:
                    break
    finally:
        del(device)



def main():
    parser = argparse.ArgumentParser(description='Acquire from multiple RealSenses.')
    parser.add_argument('-m','--mouse', type=str, default=name,
        help='ID of mouse for file naming.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Delete local dirs or not. 0=don''t delete')
    parser.add_argument('--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')

    args = parser.parse_args()

    
    # Configure depth and color streams
    # pipeline = rs.pipeline()
    config = rs.config()
    resolution_width = 480
    resolution_height = 270
    framerate = 90
    config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    serials = enumerate_connected_devices(rs.context())

    if os.path.isfile('serials.yaml'):
        with open('serials.yaml') as f:
            serial_dict = yaml.load(f)
    else:
        warnings.Warn('Need to create a lookup table for matching serial numbers with names.')
        serial_dict = {}
        for serial in serials:
            serial_dict[serial] = None

    # Start streaming
    devices = []
    experiment = '%s_%s', %(args.mouse, time.strftime('%y%M%d_%H%M%S', time.localtime()))
    for serial in serials:

        device = Device(config, serial, savedir=datadir, experiment=experiment,
            name=serial_dict[serial],preview=args.preview)
        device.start()
        devices.append(device)


    print('Acquiring...')
    with mp.Pool(len(devices)) as p:
        p.map(run_loop, devices)
    
    try:
        while True:
            start_t = time.perf_counter()

            frames = {}
            for (serial, device) in devices.items():
                streams = device.pipeline_profile.get_streams()
                # frameset = rs.composite_frame(rs.frame())
                frameset = device.pipeline.poll_for_frames()
                if frameset.size() == len(streams):
                    left = frameset.get_infrared_frame(1)
                    right = frameset.get_infrared_frame(2)
                    if not left or not right:
                        continue
                    left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
                    out = np.vstack((left,right))
                    time_acq = time.perf_counter() - start_t
                    if args.preview:
                        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # string = '%.4f' %(time_acq*1000)
                        string = serial
                        cv2.putText(out,string,(10,500), font, 0.5,(0,0,255),2,cv2.LINE_AA)
                        cv2.imshow(serial, out)
                        key = cv2.waitKey(1)
                        if key==27:
                            raise KeyboardInterrupt
            
            cputime = time.time()
            if args.save:
                ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
                ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
                if ret1 and ret2:
                    append_to_hdf5(f, 'left', left_jpg.squeeze())
                    append_to_hdf5(f, 'right', right_jpg.squeeze())
                    append_to_hdf5(f,'sestime', time_acq)
                    append_to_hdf5(f, 'cputime', cputime)

    except KeyboardInterrupt:
        print('User stopped acquisition.')
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        if args.save:
            f.close()
        # Stop streaming
        config.disable_all_streams()
        for serial, device in devices.items():
            device.pipeline.stop()


if __name__=='__main__':
    main()

