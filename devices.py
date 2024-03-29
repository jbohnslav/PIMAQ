import os
import subprocess as sp
import time
from threading import Thread
from queue import LifoQueue
import warnings

try:
    import pyrealsense2 as rs
except ImportError as e:
    print('pyrealsense not found, cannot acquire realsense cameras...')
    rs = None

import cv2
import numpy as np
import h5py
from utils import VideoWriter, append_to_hdf5

try:
    import PySpin
except ImportError as e:
    print('PySpin not found, can''t acquire from FLIR cameras')
    PySpin = None
if PySpin is not None:
    import pointgrey_utils as pg



class Device:
    def __init__(self, start_t=None,height=None,width=None,save=False,savedir=None,
                experiment=None, name=None,
                movie_format='hdf5',metadata_format='hdf5', uncompressed=False,
                preview=False,verbose=False, codec='MJPG'):
        # print('Initializing %s' %name)
        # self.config = config
        # self.serial = serial
        self.start_t = start_t
        self.save= save
        # self.savedir = savedir
        self.directory = os.path.join(savedir, experiment)
        # self.experiment = experiment
        self.name = name
        self.uncompressed = uncompressed
        # self.save_format=save_format
        self.preview=preview
        self.verbose = verbose
        self.started = False
        self.height = height
        self.width = width
        self.codec = codec

        if movie_format == 'hdf5':
            ending = '.h5'
        elif movie_format == 'ffmpeg':
            ending = '.mp4'
        elif movie_format == 'opencv':
            ending = '.avi'
        else:
            raise ValueError('unknown movie format: {}'.format(movie_format))
        if self.save:
            if uncompressed:
                filetype = '.png'
                movie_format = 'directory'
            else:
                filetype = '.png'

            self.writer_obj = VideoWriter(filename=os.path.join(self.directory, name + ending), 
                height=self.height, width=self.width, fps=30, verbose=self.verbose, 
                movie_format=movie_format, filetype=filetype, 
                )    

    def process(self):
        # should be overridden by all subclasses
        raise NotImplementedError    

    def start(self):
        # should be overridden by all subclasses
        raise NotImplementedError

    def initialize_preview(self):
        # cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.latest_frame = None
        self.preview_queue = LifoQueue(maxsize=5)
        self.preview_thread = Thread(target=self.preview_worker, args=(self.preview_queue,))
        self.preview_thread.daemon = True
        self.preview_thread.start()

    def preview_worker(self, queue):
        should_continue = True
        while should_continue:
            item = queue.get()
            # print(item)
            if item is None:
                if self.verbose:
                    print('Preview stop signal received')
                should_continue=False
                break
                # break
            # left, right, count = item
            frame, count = item
            # frame should be processed, so a single RGB image
            # out = np.vstack((left,right))
            h, w, c = frame.shape
            if self.save:
                frame = cv2.resize(frame, (w//2,h//2),cv2.INTER_NEAREST)
                out_height = h//2
            else:
                frame = cv2.resize(frame, (w,h),cv2.INTER_NEAREST)
                out_height = h//3*2
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    
            # string = '%.4f' %(time_acq*1000)
            string = '%s:%07d' %(self.name, count)
            cv2.putText(frame,string,(10,out_height-20), self.font, 0.5,(0,0,255),2,cv2.LINE_AA)
            self.latest_frame = frame

            queue.task_done()
        
    def initialize_metadata_saving_hdf5(self):
        # should be overridden by all subclasses
        raise NotImplementedError
        
    def write_metadata(self, framecount, timestamp, arrival_time, sestime, cputime):
        # should be overridden by all subclasses
        raise NotImplementedError

    def update_settings(self):
        # should be overridden by subclass
        raise NotImplementedError

    def stop_streaming(self):
        # should be overridden by subclass
        # print(dir(self.pipeline))
        raise NotImplementedError

    def loop(self):
        # should be overridden by subclass
        raise NotImplementedError

    def stop(self):
        # if self.preview:
        if not self.started:
            return
        self.stop_streaming()
        if self.preview:
            self.preview_queue.put(None)
            self.preview_thread.join()
            cv2.destroyWindow(self.name)
        if hasattr(self, 'writer_obj'):
            self.writer_obj.stop()
            del self.writer_obj

            # del(self.videoobj)
        if hasattr(self, 'metadata_obj'):
            # print('fileobj')
            self.metadata_obj.close()
        
        self.started = False
        print('Cam %s stopped' %self.name) 

    def __del__(self):
        try:
            self.stop()
        except BaseException as e:
            if self.verbose:
                print('Error in destructor of cam %s' %self.name)
                print(e)
            else:
                pass

class Realsense(Device):
    def __init__(self,serial,
                 start_t=None,options=None,save=False,savedir=None,experiment=None, name=None,
        movie_format='hdf5',metadata_format='hdf5', uncompressed=False,preview=False,verbose=False,
        master=False,codec='MJPG'):
        # use these options to override input width and height
        config = rs.config()
        if options is None:
            # use these defaults
            options = {}
            options['height'] = 480
            options['width'] = 640
            options['framerate'] = 60
            options['emitter_enabled'] = 1
            options['laser_power'] = 200
            options['exposure'] = 750
            options['gain'] = 16
            options['uncompressed'] = False
        # call the constructor for the superclass!
        # we'll inherit all attributes and methods from the Device class
        # have to double the width in this constructor because we're gonna save the left 
        # and right images concatenated horizontally
        super().__init__(start_t,options['height'], options['width']*2, save, savedir, experiment, name, 
                        movie_format, metadata_format, uncompressed,preview, verbose,codec)

        config.enable_stream(rs.stream.infrared, 1, options['width'], options['height'], 
            rs.format.y8, options['framerate'])
        config.enable_stream(rs.stream.infrared, 2, options['width'], options['height'], 
            rs.format.y8, options['framerate'])

        self.serial = str(serial)
        self.config = config
        self.master = master
        self.options = options
        self.verbose = verbose
        
    def process(self, left, right):
        # t0 = time.perf_counter()
        out = np.hstack((left, right))
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        # print('process t: %.6f' %( (time.perf_counter() - t0)*1000 ))
        return(out)
    
    def start(self):
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
        self.update_settings()
        if self.save:
            self.initialize_metadata_saving_hdf5()
            # self.initialize_saving()
            if self.verbose:
                print('saving initialized: %s' %self.name)
        if self.preview:
            self.initialize_preview()
        self.started= True
        
    def update_settings(self):
        # sensor = self.prof.get_device().first_depth_sensor()
        # print(dir(sensor))
        # sensor.set_option(rs.option.emitter_enabled,1)
        this_device = self.prof.get_device()
        ir_sensors = this_device.query_sensors()[0] # 1 for RGB
        # turn auto exposure off! Very important
        ir_sensors.set_option(rs.option.enable_auto_exposure,0)

        assert(self.options['emitter_enabled']==0 or self.options['emitter_enabled']==1)
        ir_sensors.set_option(rs.option.emitter_enabled, self.options['emitter_enabled'])
        laser_pwr = ir_sensors.get_option(rs.option.laser_power)
        if self.verbose:
            print("laser power = ", laser_pwr)
        laser_range = ir_sensors.get_option_range(rs.option.laser_power)
        if self.verbose:
            print("laser power range = " , laser_range.min , "~", laser_range.max)
        assert(self.options['laser_power']<=laser_range.max and self.options['laser_power']>=laser_range.min)
        ir_sensors.set_option(rs.option.laser_power,self.options['laser_power'])

        
        ir_sensors.set_option(rs.option.exposure, self.options['exposure'])
        gain_range = ir_sensors.get_option_range(rs.option.gain)
        if self.verbose:
            print("gain range = " , gain_range.min , "~", gain_range.max)
        assert(self.options['gain']<=gain_range.max and self.options['gain']>=gain_range.min)
        ir_sensors.set_option(rs.option.gain,self.options['gain'])

        if self.master:
            mode = 1
        else:
            mode = 2
        if self.verbose:
            print('%s: %s,%d' %(self.name, 'master' if self.master else 'slave', mode))
        # set this to 2 for slave mode, 1 for master!
        ir_sensors.set_option(rs.option.inter_cam_sync_mode, mode)
        # print('sync mode ', ir_sensors.get_option(rs.option.inter_cam_sync_mode))
    def loop(self):
        if not hasattr(self, 'pipeline'):
            raise ValueError('Start must be called before loop!')
        N_streams =  len(self.prof.get_streams())

        try:
            should_continue = True
            while should_continue:
                
                # absolutely essential to use poll_for_frames rather than wait_for_frames!
                # it might work if you run loop in a subthread
                # but if you use wait_for_frames in the main thread, it will block execution
                # of subthreads, like writing to disk, etc
                frames = self.pipeline.poll_for_frames()
                if frames.size() == N_streams:
                    pass
                else:
                    time.sleep(1/400)
                    continue
                
                start_t = time.perf_counter()
                left = frames.get_infrared_frame(1)
                right = frames.get_infrared_frame(2)
                if not left or not right:
                    continue
                left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())
                frame = self.process(left, right)
                sestime = time.perf_counter() - self.start_t
                cputime = time.time()
                framecount = frames.get_frame_number()
                # by default, milliseconds from 1970. convert to seconds for datetime.datetime
                arrival_time = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)/1000
                timestamp = frames.get_timestamp()/1000
                
                # print('standard process time: %.6f' %(time.perf_counter() - start_t))
                # def write_metadata(self, framecount, timestamp, arrival_time, sestime, cputime):
                metadata = (framecount, timestamp, arrival_time, sestime, cputime)
                if self.save:
                    self.writer_obj.write(frame)
                    self.write_metadata(framecount, timestamp, arrival_time, sestime, cputime)
                    # self.save_queue.put_nowait((frame, metadata))

                # only output every 10th frame for speed
                # might be unnecessary
                if self.preview and framecount % 10 ==0:
                    self.preview_queue.put_nowait((frame,framecount))
                    if self.latest_frame is not None:
                        cv2.imshow(self.name, self.latest_frame)
                        key = cv2.waitKey(1)
                        if key==27:
                            break
                frames = None
            
        except KeyboardInterrupt:
            print('keyboard interrupt')
            should_continue = False
        finally:
            # don't know why I can't put this in the destructor
            # print(dir(device))
            # if device.preview:
            #     device.preview_queue.put(None)
            #     device.preview_thread.join()
            # time.sleep(1)
            self.stop()

    def initialize_metadata_saving_hdf5(self):
        fname = os.path.join(self.directory, self.name + '_metadata.h5')
        f = h5py.File(fname, 'w')
        
        dset = f.create_dataset('framecount',(0,),maxshape=(None,),dtype=np.int32)
        dset = f.create_dataset('timestamp',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('arrival_time',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('sestime',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('cputime',(0,),maxshape=(None,),dtype=np.float64)
        
        intrinsics, extrinsics = self.get_intrinsic_extrinsic()
        dset = f.create_dataset('intrinsics', data=intrinsics)
        # print(list(f.keys()))
        # f['intrinsics'] = intrinsics
        dset = f.create_dataset('extrinsics', data=extrinsics)
        # f['extrinsics'] = extrinsics

        self.metadata_obj = f
        
    def write_metadata(self, framecount, timestamp, arrival_time, sestime, cputime):
        # t0 = time.perf_counter()
        append_to_hdf5(self.metadata_obj,'framecount', framecount)
        append_to_hdf5(self.metadata_obj,'timestamp', timestamp)
        append_to_hdf5(self.metadata_obj,'arrival_time', arrival_time)
        append_to_hdf5(self.metadata_obj,'sestime', sestime)
        append_to_hdf5(self.metadata_obj, 'cputime', cputime)

    def get_intrinsic_extrinsic(self):
        intrinsics = self.prof.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_intrinsics()
        K = np.zeros((3,3),dtype=np.float64)
        K[0,0] = intrinsics.fx
        K[0,2] = intrinsics.ppx
        K[1,1] = intrinsics.fy
        K[1,2] = intrinsics.ppy
        K[2,2] = 1

        extrinsics = self.prof.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_extrinsics_to(self.prof.get_stream(
            rs.stream.infrared,2))
        # print(extrinsics.rotation)
        R = np.array(extrinsics.rotation).reshape(3,3)
        t = np.array(extrinsics.translation)

        extrinsic = np.zeros((3,4), dtype=np.float64)
        extrinsic[:3,:3] = R
        extrinsic[:3,3] = t
        return(K, extrinsic)


    def stop_streaming(self):
        # print(dir(self.pipeline))
        try:
            self.pipeline.stop()
            self.config.disable_all_streams()
        except BaseException as e:
            if self.verbose:
                print('Probably tried to call stop before a start.')
                print(e)
            else:
                pass


class PointGrey(Device):
    def __init__(self,serial,
                 start_t=None,options=None,save=False,savedir=None,experiment=None, name=None,
        movie_format='opencv',metadata_format='hdf5', uncompressed=False,preview=False,verbose=False,
        strobe=None,codec='MJPG'):

        self.verbose=verbose
        # use these options to override input width and height
        # Retrieve singleton reference to system object
        system = PySpin.System.GetInstance()

        # now that we have width and height, call the constructor for the superclass!
        # we'll inherit all attributes and methods from the Device class
        # have to double the width in this constructor because we're gonna save the left 
        # and right images concatenated horizontally
        super().__init__(start_t,options['Height'], options['Width'], save, savedir, experiment, name, 
                        movie_format, metadata_format, uncompressed,preview, verbose,codec)

        version = system.GetLibraryVersion()
        
        if self.verbose:
            print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()
        if verbose:
            print('Number of cameras detected: %d' % num_cameras)

        self.cam = None
        for c in cam_list:
            this_serial = pg.get_serial_number(c)
            if int(this_serial) == int(serial):
                self.cam = c
        if self.cam is None:
            raise ValueError('Didn''t find serial! %s' %serial)
        cam_list.Clear()
        self.cam.Init()
        self.compute_timestamp_offset()

        self.nodemap = self.cam.GetNodeMap()
        self.serial = serial
        # self.cam = cam
        self.system = system
        if self.save:
            self.initialize_metadata_saving_hdf5()
        

        if options is None:
            # Note: modify these at your own risk! Don't change the order!
            # many have dependencies that are hard to figure out, so the order matters.
            # For example, ExposureAuto must be set to Off before ExposureTime can be changed. 
            options = {
                  'AcquisitionMode': 'Continuous', # can capture one frame or multiple frames as well
                  'ExposureAuto': 'Off', # manually set exposure
                  'ExposureTime': 1000.0, # in microseconds, so 1000 = 1ms
                  # this downsamples image in half, enabling faster framerates
                  # it's not possible to change BinningHorizontal, but it is automatically changed by changing
                  # BinningVertical
                  'BinningVertical': 2, 
                  'Height': 512, # max 1024 if Binning=1, else 512
                  'Width': 640, # max 1280 if Binning=1, else 640
                  'OffsetX': 0, # left value of ROI
                  'OffsetY': 0, # right value of ROI
                  'PixelFormat': 'Mono8',
                  'AcquisitionFrameRateAuto': 'Off',
                  'AcquisitionFrameRate': 60.0,
                  'GainAuto': 'Off',
                  'Gain': 10.0,
                  'SharpnessAuto': 'Off'}
        if strobe is None:
            strobe = {
            'line': 2,
            'duration': 0.0
            }
        self.options = options
        self.strobe = strobe

    def compute_timestamp_offset(self):
        self.cam.TimestampLatch.Execute()
        self.timestamp_offset = time.perf_counter() - self.cam.TimestampLatchValue.GetValue()*1e-9 - self.start_t

    def process(self, frame):
        # t0 = time.perf_counter()
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # print('process t: %.6f' %( (time.perf_counter() - t0)*1000 ))
        return(out)
    
    def start(self, sync_mode=None):

        self.update_settings()
        self.cam.BeginAcquisition()
        if self.save:
            # self.initialize_saving()
            print('saving initialized: %s' %self.name)
        if self.preview:
            self.initialize_preview()
        self.started= True
        
    def update_settings(self):
        """ Updates PointGrey camera settings.
        Attributes, types, and range of possible values for each attribute are available
        in the camera documentation. 
        These are extraordinarily tricky! Order matters! For exampple, ExposureAuto must be set
        to Off before ExposureTime can be set. 
        """
        for key, value in self.options.items():
            pg.set_value(self.nodemap, key, value)
        # changing strobe involves multiple variables in the correct order, so I've bundled
        # them into this function
        pg.turn_strobe_on(self.nodemap, self.strobe['line'], strobe_duration=self.strobe['duration'])

    def loop(self):
        if not self.started:
            raise ValueError('Start must be called before loop!')
        
        try:
            should_continue = True
            while should_continue:
                image_result = self.cam.GetNextImage()
                if image_result.IsIncomplete():
                    if self.verbose:
                        print('Image incomplete with image status %d ...' 
                            % image_result.GetImageStatus())
                    continue
                else:
                    pass
                image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, 
                    PySpin.HQ_LINEAR)
                frame = image_converted.GetNDArray()
                
                frame = self.process(frame)
                sestime = time.perf_counter() - self.start_t
                cputime = time.time()
                framecount =  image_result.GetFrameID()
                # timestamp is nanoseconds from last time camera was powered off
                timestamp = image_result.GetTimeStamp()*1e-9 + self.timestamp_offset
                
                # print('standard process time: %.6f' %(time.perf_counter() - start_t))
                # def write_metadata(self, framecount, timestamp, arrival_time, sestime, cputime):
                # metadata = (framecount, timestamp, sestime, cputime)
                if self.save:
                    self.writer_obj.write(frame)
                    self.write_metadata(framecount, timestamp, sestime, cputime)
                    # self.save_queue.put_nowait((frame, metadata))

                # only output every 10th frame for speed
                # might be unnecessary
                if self.preview and framecount % 10 ==0:
                    self.preview_queue.put_nowait((frame,framecount))
                    if self.latest_frame is not None:
                        cv2.imshow(self.name, self.latest_frame)
                        key = cv2.waitKey(1)
                        if key==27:
                            break
                frames = None
            
        except KeyboardInterrupt:
            print('keyboard interrupt')
            should_continue = False
        finally:
            # don't know why I can't put this in the destructor
            # print(dir(device))
            # if device.preview:
            #     device.preview_queue.put(None)
            #     device.preview_thread.join()
            # time.sleep(1)
            self.stop()

    def initialize_metadata_saving_hdf5(self):
        fname = os.path.join(self.directory, self.name + '_metadata.h5')
        f = h5py.File(fname, 'w')
        
        dset = f.create_dataset('framecount',(0,),maxshape=(None,),dtype=np.int32)
        dset = f.create_dataset('timestamp',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('sestime',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('cputime',(0,),maxshape=(None,),dtype=np.float64)

        self.metadata_obj = f
        
    def write_metadata(self, framecount, timestamp,  sestime, cputime):
        # t0 = time.perf_counter()
        append_to_hdf5(self.metadata_obj,'framecount', framecount)
        append_to_hdf5(self.metadata_obj,'timestamp', timestamp)
        append_to_hdf5(self.metadata_obj,'sestime', sestime)
        append_to_hdf5(self.metadata_obj, 'cputime', cputime)


    def stop_streaming(self):
        # print(dir(self.pipeline))
        try:
            self.cam.EndAcquisition()
            del(self.nodemap)
            self.cam.DeInit()
            del(self.cam)
            self.system.ReleaseInstance()
        except BaseException as e:
            if self.verbose:
                print('Probably tried to call stop before a start.')
                print(e)
            else:
                pass