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
    def __init__(self, serial,save=False,savedir=None,experiment=None, name=None,
        save_format='hdf5',preview=False):
        # print('Initializing %s' %name)
        # self.config = config
        self.serial = serial
        self.save= save
        self.savedir = savedir
        self.experiment = experiment
        self.name = name
        self.save_format=save_format
        self.preview=preview
        # print('Done.')

    def start(self, config):
        pipeline = rs.pipeline()
        config.enable_device(self.serial)
        try:
            pipeline_profile = pipeline.start(config)
        except RuntimeError:
            print('Pipeline for camera %s already running, restarting...' %serial)
            pipeline.stop()
            time.sleep(1)
            pipeline_profile = pipeline.start(config)
        self.pipeline = pipeline
        self.prof = pipeline_profile
        time.sleep(1)
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
        # sensor = self.prof.get_device().first_depth_sensor()
        # print(dir(sensor))
        # sensor.set_option(rs.option.emitter_enabled,1)
        
        this_device = self.prof.get_device()
        ir_sensors = this_device.query_sensors()[0] # 1 for RGB
        ir_sensors.set_option(rs.option.emitter_enabled,1)
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
        if hasattr(self, 'pipeline'):
            self.stop_streaming()
        if self.save:
            self.fileobj.close()
        if self.preview:
            cv2.destroyWindow(self.name)
        print('Destructor called, cam %s deleted.' %self.name) 

# def initialize_and_loop()

def run_loop(device):
    start_t = time.perf_counter()
    framecount = 0
    if device.preview:
        font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            # print('acquiring')
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
                string = '%s:%07d' %(device.name, framecount)
                cv2.putText(out,string,(10,500), font, 0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow(device.name, out)
                key = cv2.waitKey(1)
                if key==27:
                    break
            framecount+=1
    except KeyboardInterrupt:
        print('User stopped acquisition.')
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
    experiment = '%s_%s' %(args.mouse, time.strftime('%y%M%d_%H%M%S', time.localtime()))
    print('initializing devices')

    for serial in serials:
        device = Device(serial, save=args.save,savedir=datadir, experiment=experiment,
            name=serial_dict[serial],preview=args.preview)
        # time.sleep(1)
        device.start(config)
        devices.append(device)
    print('done')

    # run_loop(devices[0])
    # print('Acquiring...')
    # processes = []
    # for device in devices:
    #     p = mp.Process(target=run_loop, args=(device,))
    #     processes.append(p)
    # print('Starting')
    # for p in processes:
    #     p.start()
    # # [p.start() for p in processes]
    # for p in processes:
    #     p.join()
    # print('Done')
    # for p in processes:
    #     p.terminate()

    # [p.join() for p in processes]
    # for p in proc:
    #     p.start()

    # with mp.Pool(len(devices)) as p:
    #     p.map(run_loop, devices)
    
    if args.preview:
        cv2.destroyAllWindows()
    # Stop streaming
    config.disable_all_streams()

if __name__=='__main__':
    main()

