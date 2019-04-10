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
from queue import LifoQueue
from threading import Thread

datadir = r'D:\DATA\JB\realsense'
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
        save_format='hdf5',preview=False,verbose=False,options=None):
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
        self.verbose = verbose
        self.options = options
        # print('Done.')

    def start(self, sync_mode='slave'):
        pipeline = rs.pipeline()
        self.config.enable_device(self.serial)
        try:
            pipeline_profile = pipeline.start(self.config)
        except RuntimeError:
            print('Pipeline for camera %s already running, restarting...' %self.serial)
            pipeline.stop()
            time.sleep(1)
            pipeline_profile = pipeline.start(self.config)
        CAPACITY = 10
        # print(dir(pipeline))
        # self.framequeue = rs.framequeue(CAPACITY)
        self.pipeline = pipeline
        self.prof = pipeline_profile
        time.sleep(1)
        self.update_settings(sync_mode)
        if self.save:
            self.initialize_saving()
            print('saving initialized: %s' %self.name)
        if self.preview:
            self.initialize_preview()

    def initialize_preview(self):
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.latest_frame = None
        self.preview_queue = LifoQueue(maxsize=5)
        self.preview_thread = Thread(target=self.preview_worker, args=(self.preview_queue,))
        self.preview_thread.daemon = True
        self.preview_thread.start()

    def preview_worker(self, queue):
        while True:
            item = queue.get()
            # print(item)
            if item is None:
                # print('Stop signal received')
                break
            left, right, count = item
            out = np.vstack((left,right))
            h, w = out.shape
            if self.save:
                out = cv2.resize(out, (w//2,h//2),cv2.INTER_NEAREST)
                out_height = h//2
            else:
                out = cv2.resize(out, (w//3*2,h//3*2),cv2.INTER_NEAREST)
                out_height = h//3*2
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                    
            # string = '%.4f' %(time_acq*1000)
            string = '%s:%07d' %(self.name, count)
            cv2.putText(out,string,(10,out_height-20), self.font, 0.5,(0,0,255),2,cv2.LINE_AA)
            self.latest_frame = out

            queue.task_done()

    def initialize_saving(self):
        if self.save_format == 'hdf5':
            # fname = '%s_%s.h5' %(self.experiment, self.name)
            fname = self.name + '.h5'
            subdir = os.path.join(self.savedir, self.experiment)
            if not os.path.isdir(subdir):
                os.makedirs(subdir)
            # fname = self.experiment + '.h5'
            f = h5py.File(os.path.join(subdir, fname), 'w')
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
        print(subdir)

    def update_settings(self, sync_mode='master'):
        # sensor = self.prof.get_device().first_depth_sensor()
        # print(dir(sensor))
        # sensor.set_option(rs.option.emitter_enabled,1)
        if sync_mode=='master':
            mode = 1
        elif sync_mode == 'slave':
            mode = 2
        if self.verbose:
            print('%s: %s,%d' %(self.name, sync_mode, mode))
        
        this_device = self.prof.get_device()
        ir_sensors = this_device.query_sensors()[0] # 1 for RGB
        if self.options=='default' or self.options=='large':
            ir_sensors.set_option(rs.option.emitter_enabled,1)
            ir_sensors.set_option(rs.option.enable_auto_exposure,0)
            laser_pwr = ir_sensors.get_option(rs.option.laser_power)
            if self.verbose:
                print("laser power = ", laser_pwr)
            laser_range = ir_sensors.get_option_range(rs.option.laser_power)
            if self.verbose:
                print("laser power range = " , laser_range.min , "~", laser_range.max)
            ir_sensors.set_option(rs.option.laser_power,300)
            ir_sensors.set_option(rs.option.exposure,650)
            ir_sensors.set_option(rs.option.gain,16)

        elif self.options =='calib':
            ir_sensors.set_option(rs.option.emitter_enabled,0)
            ir_sensors.set_option(rs.option.enable_auto_exposure,0)
            ir_sensors.set_option(rs.option.exposure,1500)
            ir_sensors.set_option(rs.option.gain,16)
            self.jpg_quality = 99

        if self.options=='large':
            ir_sensors.set_option(rs.option.exposure,750)
            ir_sensors.set_option(rs.option.laser_power,200)
            ir_sensors.set_option(rs.option.gain,16)
        if self.options=='brighter':
            gain_range = ir_sensors.get_option_range(rs.option.gain)
            if self.verbose:
                print("gain range = " , gain_range.min , "~", gain_range.max)
            ir_sensors.set_option(rs.option.exposure, 500)
            ir_sensors.set_option(rs.option.gain,16)
        # set this to 2 for slave mode, 1 for master!
        ir_sensors.set_option(rs.option.inter_cam_sync_mode, mode)
        # print(ir_sensors.supports(rs.option.inter_cam_sync_mode))
        # this_device.set_option(rs.option.inter_cam_sync_mode,2)
        # from github
        # ir_sensors.set_option(rs.option.frames_queue_size,7)
        # print(ir_sensors.get_option(rs.option.inter_cam_sync_mode))

    def write_frames(self,left,right, framecount, timestamp,
                        arrival_time,sestime,cputime):
        if not hasattr(self, 'fileobj'):
            raise ValueError('Writing for camera %s not initialized.' %self.camname)
        # ret1, ret2 = True, True
        ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,self.jpg_quality))
        ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,self.jpg_quality))
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
        # if self.preview:

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
    if args.options=='default' or args.options=='brighter':
        resolution_width = 480
        resolution_height = 270
        framerate = 90
    elif args.options=='large':
        resolution_width = 640
        resolution_height = 480
        framerate=60
    elif args.options=='calib':
        resolution_width=640
        resolution_height=480
        framerate=6
    else:
        raise NotImplementedError
    config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
    device = Device(serial, config, start_t,save=args.save,savedir=datadir, experiment=experiment,
            name=serial_dict[serial],preview=args.preview,verbose=args.verbose, options=args.options)
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
    # framecount = 0
   #  CAPACITY = 100
    # framequeue = rs.frame_queue(CAPACITY)
    # print(dir(framequeue))
    try:
        while True:
            # print('acquiring')
            frames = device.pipeline.wait_for_frames(1000*10)
            # frames =  rs.composite_frame(rs.frame())
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
            sestime = time.perf_counter() - device.start_t
            cputime = time.time()
            framecount = frames.get_frame_number()
            # by default, milliseconds from 1970. convert to seconds for datetime.datetime
            arrival_time = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)/1000
            timestamp = frames.get_timestamp()/1000
            if device.save:
                device.write_frames(left, right, framecount, timestamp,
                    arrival_time, sestime, cputime)
                # if saving, be more stringent about previewing
                # condition = (time.perf_counter()-start_t)*1000<8 and framecount%5==0
                condition = (time.perf_counter()-start_t)*1000<50 and framecount%5==0
            else:
                condition = True

            # print(time.perf_counter()-start_t)
            if device.preview and condition:
                # print(time.perf_counter()-start_t)
                device.preview_queue.put((left,right,framecount))
                if device.latest_frame is not None:
                    cv2.imshow(device.name, device.latest_frame)
                    key = cv2.waitKey(1)
                    if key==27:
                        break
                # out = np.vstack((left,right))
                # out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                        
                # # string = '%.4f' %(time_acq*1000)
                # string = '%s:%07d' %(device.name, framecount)
                # cv2.putText(out,string,(10,500), font, 0.5,(0,0,255),2,cv2.LINE_AA)
                # cv2.imshow(device.name, out)
                # key = cv2.waitKey(1)
                # if key==27:
                #     break
            # framecount+=1
    except KeyboardInterrupt:
        pass
        # print('User stopped acquisition.')
    finally:
        # don't know why I can't put this in the destructor
        if device.preview:
            device.preview_queue.put(None)
            device.preview_thread.join()
        del(device)

def main():
    parser = argparse.ArgumentParser(description='Acquire from multiple RealSenses.')
    parser.add_argument('-m','--mouse', type=str, default=name,
        help='ID of mouse for file naming.')
    parser.add_argument('-p', '--preview', default=False, action='store_true',
        help='Show preview in opencv window')
    parser.add_argument('-s', '--save', default=False, action='store_true',
        help='Delete local dirs or not. 0=don''t delete')
    parser.add_argument('-v', '--verbose', default=False,action='store_true',
        help='Use this flag to print debugging commands.')
    parser.add_argument('--master', default=master,type=str,
        help='Which camera serial number is the "master" camera.')
    parser.add_argument('-o','--options', default='default',
        choices=['default','large', 'calib', 'brighter'], type=str,
        help='Which camera serial number is the "master" camera.')

    args = parser.parse_args()

    # Configure depth and color streams
    # pipeline = rs.pipeline()
    
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    serials = enumerate_connected_devices(rs.context())
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

