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
master = '830112071475'

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
    def __init__(self, serial,config, start_t, save=False,savedir=None,experiment=None, name=None,
        save_format='hdf5',preview=False):
        # print('Initializing %s' %name)
        self.config = config
        self.serial = serial
        self.start_t = start_t
        self.save= save
        self.savedir = savedir
        self.experiment = experiment
        self.name = name
        self.save_format=save_format
        self.preview=preview
        # print('Done.')

    def start(self, sync_mode='slave'):
        pipeline = rs.pipeline()
        self.config.enable_device(self.serial)
        try:
            pipeline_profile = pipeline.start(self.config)
        except RuntimeError:
            print('Pipeline for camera %s already running, restarting...' %serial)
            pipeline.stop()
            time.sleep(1)
            pipeline_profile = pipeline.start(self.config)
        self.pipeline = pipeline
        self.prof = pipeline_profile
        time.sleep(1)
        self.update_settings(sync_mode)
        if self.save:
            self.initialize_saving()
        if self.preview:
            self.initialize_preview()

    def initialize_preview(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)

    def initialize_saving(self):
        if self.save_format == 'hdf5':
            fname = '%s_%s.h5' %(self.experiment, self.name)
            # fname = self.experiment + '.h5'
            f = h5py.File(os.path.join(self.savedir, fname), 'w')
            datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
            dset = f.create_dataset('left', (0,), maxshape=(None,),dtype=datatype)
            dset = f.create_dataset('right', (0,), maxshape=(None,),dtype=datatype)
            dset = f.create_dataset('framecount',(0,),maxshape=(None,),dtype=np.int32)
            dset = f.create_dataset('timestamp',(0,),maxshape=(None,),dtype=np.float64)
            dset = f.create_dataset('arrival_time',(0,),maxshape=(None,),dtype=np.float64)
            dset = f.create_dataset('sestime',(0,),maxshape=(None,),dtype=np.float64)
            dset = f.create_dataset('cputime',(0,),maxshape=(None,),dtype=np.float64)
            self.fileobj = f
        else:
            raise NotImplementedError

    def update_settings(self, sync_mode='master'):
        # sensor = self.prof.get_device().first_depth_sensor()
        # print(dir(sensor))
        # sensor.set_option(rs.option.emitter_enabled,1)
        if sync_mode=='master':
            mode = 1
        elif sync_mode == 'slave':
            mode = 2
        
        this_device = self.prof.get_device()
        ir_sensors = this_device.query_sensors()[0] # 1 for RGB
        ir_sensors.set_option(rs.option.emitter_enabled,1)
        ir_sensors.set_option(rs.option.enable_auto_exposure,0)
        ir_sensors.set_option(rs.option.exposure,500)
        ir_sensors.set_option(rs.option.gain,16)
        # set this to 2 for slave mode, 1 for master!
        ir_sensors.set_option(rs.option.inter_cam_sync_mode, mode)
        # print(ir_sensors.supports(rs.option.inter_cam_sync_mode))
        # this_device.set_option(rs.option.inter_cam_sync_mode,2)
        # from github
        # ir_sensors.set_option(rs.option.frames_queue_size,1)
        # print(ir_sensors.get_option(rs.option.inter_cam_sync_mode))

    def write_frames(self,left,right, framecount, timestamp,
                        arrival_time,sestime,cputime):
        if not hasattr(self, 'fileobj'):
            raise ValueError('Writing for camera %s not initialized.' %self.camname)
        ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
        ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
        if ret1 and ret2:
            append_to_hdf5(self.fileobj, 'left', left_jpg.squeeze())
            append_to_hdf5(self.fileobj, 'right', right_jpg.squeeze())
            append_to_hdf5(self.fileobj,'framecount', framecount)
            append_to_hdf5(self.fileobj,'timestamp', timestamp)
            append_to_hdf5(self.fileobj,'arrival_time', arrival_time)
            append_to_hdf5(self.fileobj,'sestime', sestime)
            append_to_hdf5(self.fileobj, 'cputime', cputime)

    def stop_streaming(self):
        self.pipeline.stop()
        self.config.disable_all_streams()

    def __del__(self):
        if hasattr(self, 'pipeline'):

            self.stop_streaming()
        if self.save:
            self.fileobj.close()
        if self.preview:
            cv2.destroyWindow(self.name)
        print('Destructor called, cam %s deleted.' %self.name) 

def initialize_and_loop(serial,args,datadir, experiment, serial_dict,start_t):
    config = rs.config()
    resolution_width = 480
    resolution_height = 270
    framerate = 90
    config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
    device = Device(serial, config, start_t,save=args.save,savedir=datadir, experiment=experiment,
            name=serial_dict[serial],preview=args.preview)
    assert(type(args.master)==str)
    master = True if serial == args.master else False
    sync_mode = 'master' if master else 'slave'
    device.start(sync_mode=sync_mode)
    if not master:
        # this makes sure that the master camera is the first one to start
        time.sleep(3)

    run_loop(device)

def run_loop(device):
    # start_t = time.perf_counter()
    framecount = 0
    if device.preview:
        font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            # print('acquiring')
            frames = device.pipeline.wait_for_frames(1000*10)
            # frames = device.pipeline.poll_for_frames()
            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
            sestime = time.perf_counter() - device.start_t
            cputime = time.time()
            framecount = frames.get_frame_number()
            # by default, milliseconds from 1970. convert to seconds for datetime.datetime
            arrival_time = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)/1000
            timestamp = frames.get_timestamp()/1000
            if device.save:
                device.write_frames(left, right, framecount, timestamp,
                    arrival_time, sestime, cputime)

            if device.preview:
                out = np.vstack((left,right))
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                        
                # string = '%.4f' %(time_acq*1000)
                string = '%s:%07d' %(device.name, framecount)
                cv2.putText(out,string,(10,500), font, 0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow(device.name, out)
                key = cv2.waitKey(1)
                if key==27:
                    break
            framecount+=1
    except KeyboardInterrupt:
        pass
        # print('User stopped acquisition.')
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
    parser.add_argument('--master', default=master,type=str,
        help='Which camera serial number is the "master" camera.')

    args = parser.parse_args()

    
    # Configure depth and color streams
    # pipeline = rs.pipeline()
    
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    serials = enumerate_connected_devices(rs.context())
    assert(args.master in serials)
    if os.path.isfile('serials.yaml'):
        with open('serials.yaml') as f:
            serial_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        warnings.Warn('Need to create a lookup table for matching serial numbers with names.')
        serial_dict = {}
        for serial in serials:
            serial_dict[serial] = None

    # Start streaming
    experiment = '%s_%s' %(args.mouse, time.strftime('%y%M%d_%H%M%S', time.localtime()))

    """
    Originally I wanted to initialize each device, then pass each device to "run_loop" 
    in its own process. However, pyrealsense2 config objects and pyrealsense2 pipeline objects
    are not pickle-able, and python pickles arguments before passing them to a process. Therefore,
    you have to initialize the configuration and the pipeline from within the process already!
    """
    start_t = time.perf_counter()
    tuples = []
    for serial in serials:
        tup = (serial, args, datadir, experiment, serial_dict,start_t)
        tuples.append(tup)
    with mp.Pool(len(serials)) as p:
        try:
            p.starmap(initialize_and_loop, tuples)  
        except KeyboardInterrupt:
            print('User interrupted acquisition')

    
    if args.preview:
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
