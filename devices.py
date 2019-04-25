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
import subprocess as sp

def initialize_hdf5(filename, framesize=None, codec=None):
    filename = filename + '.h5'
    f = h5py.File(filename, 'w')
    datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset('frame', (0,), maxshape=(None,),dtype=datatype)
    # dset = f.create_dataset('right', (0,), maxshape=(None,),dtype=datatype)
    return(f)

def write_frame_hdf5(writer_obj, frame, axis=0):
    # ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
    # ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
    ret, jpg = cv2.imencode('.jpg', frame, (cv2.IMWRITE_JPEG_QUALITY,80))
    writer_obj['frame'].resize(writer_obj['frame'].shape[axis]+1, axis=axis)
    # f['left'].resize(f['left'].shape[axis]+1, axis=axis)
    writer_obj['frame'][-1]=jpg.squeeze()
     
def initialize_opencv(filename, framesize, codec):
    filename = filename + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(filename,fourcc, 30, framesize)
    return(writer)
def write_frame_opencv(writer_obj, frame):
    # out = cv2.cvtColor(np.hstack((left, right)), cv2.COLOR_GRAY2RGB)
    # t0 = time.perf_counter()
    writer_obj.write(frame)
    # print('image writing t: %.6f' %( (time.perf_counter() - t0)*1000 ))
    
def initialize_ffmpeg(filename,framesize, codec=None):
    filename = filename + '.avi'
    size_string = '%dx%d' %framesize
    # outname = os.path.join(outdir, fname)
    command = [ 'ffmpeg',
        '-threads', '1',
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', size_string, # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '30', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
        '-crf', '23', 
        filename]
    # if you want to print to the command line, change stderr to sp.STDOUT
    pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.DEVNULL)
    return(pipe)
# from here 
# https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
def write_frame_ffmpeg(pipe, frame):
    # out = cv2.cvtColor(np.hstack((left,right)), cv2.COLOR_GRAY2RGB)
    # t0 = time.perf_counter()
    try:
        pipe.stdin.write(frame.tobytes())
    except BaseException as err:
        _, ffmpeg_error = pipe.communicate()
        error = (str(err) + ("\n\nerror: FFMPEG encountered "
                             "the following error while writing file:"
                             "\n\n %s" % (str(ffmpeg_error))))
    # print('image writing t: %.6f' %( (time.perf_counter() - t0)*1000 ))
        
def append_to_hdf5(f, name, value, axis=0):
    f[name].resize(f[name].shape[axis]+1, axis=axis)
    f[name][-1]=value

class Device:
    def __init__(self, start_t=None,height=None,width=None,save=False,savedir=None,
                 experiment=None, name=None,
        movie_format='hdf5',metadata_format='hdf5', preview=False,verbose=False,options=None):
        # print('Initializing %s' %name)
        # self.config = config
        # self.serial = serial
        self.start_t = start_t
        self.save= save
        self.savedir = savedir
        self.experiment = experiment
        self.name = name
        # self.save_format=save_format
        self.preview=preview
        self.verbose = verbose
        self.options = options
        self.started = False
        self.height = height
        self.width = width
        
        assert(movie_format in ['hdf5', 'opencv', 'ffmpeg'])
        assert(metadata_format in ['hdf5', 'csv'])
        self.movie_format = movie_format
        self.metadata_format = metadata_format
        
        if movie_format == 'hdf5':
            self.initialization_func = initialize_hdf5
            self.write_frame = write_frame_hdf5
        elif movie_format == 'opencv':
            self.initialization_func = initialize_opencv
            self.write_frame = write_frame_opencv
        elif movie_format == 'ffmpeg':
            self.initialization_func = initialize_ffmpeg
            self.write_frame = write_frame_ffmpeg
    
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
        fname = os.path.join(self.directory, self.name + '_metadata.h5')
        f = h5py.File(fname, 'w')
        
        dset = f.create_dataset('framecount',(0,),maxshape=(None,),dtype=np.int32)
        dset = f.create_dataset('timestamp',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('arrival_time',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('sestime',(0,),maxshape=(None,),dtype=np.float64)
        dset = f.create_dataset('cputime',(0,),maxshape=(None,),dtype=np.float64)
        
        self.metadata_obj = f
        
    def write_metadata(self, framecount, timestamp, arrival_time, sestime, cputime):
        # t0 = time.perf_counter()
        append_to_hdf5(self.metadata_obj,'framecount', framecount)
        append_to_hdf5(self.metadata_obj,'timestamp', timestamp)
        append_to_hdf5(self.metadata_obj,'arrival_time', arrival_time)
        append_to_hdf5(self.metadata_obj,'sestime', sestime)
        append_to_hdf5(self.metadata_obj, 'cputime', cputime)
        # print('metadata writing t: %.6f' %( (time.perf_counter() - t0)*1000 ))

    def save_worker(self, queue):
        should_continue = True
        while True:
            try:
                item = queue.get()
                # print(item)
                if item is None:
                    if self.verbose:
                        print('Saver stop signal received')
                    should_continue = False
                    break
                # left, right, count = item
                frame, metadata = item

                self.write_frame(self.writer_obj, frame)
                self.write_metadata(*metadata)

                
                print(queue.qsize())
                # time.sleep(1/120)
            except Exception as e:
                print(e)
            finally:
                queue.task_done()
        
        print('out of save queue')
        
    def initialize_saving(self):
        assert(self.savedir is not None and self.experiment is not None)
        directory = os.path.join(self.savedir, self.experiment)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.directory = directory
        
        if self.metadata_format == 'hdf5':
            self.initialize_metadata_saving_hdf5()
        else:
            raise NotImplementedError
        
        framesize = (self.width, self.height)
        codec = 'DIVX'
        filename = os.path.join(self.directory, self.name)
        writer_obj = self.initialization_func(filename, framesize, codec)
        self.writer_obj = writer_obj
        
        self.save_queue = Queue(maxsize=600)
        self.save_thread = Thread(target=self.save_worker, args=(self.save_queue,))
        self.save_thread.daemon = True
        self.save_thread.start()

    def update_settings(self, sync_mode='master'):
        # should be overridden by subclass
        raise NotImplementedError

    def stop_streaming(self):
        # should be overridden by subclass
        # print(dir(self.pipeline))
        raise NotImplementedError

    def stop(self):
        # if self.preview:
        if not self.started:
            return
        if self.save:
            print('Waiting for saving thread to finish on cam %s. DO NOT INTERRUPT' %self.name)
            self.save_queue.put(None)
            print('joining...')
            self.save_queue.join()
            print('joined')
        if self.preview:
            self.preview_queue.put(None)
            self.preview_thread.join()
            cv2.destroyWindow(self.name)
        if hasattr(self, 'pipeline'):
            # print('stream')
            self.stop_streaming()
        if hasattr(self, 'writer_obj'):
            # print('videoobj')
            if self.movie_format == 'opencv':
                self.writer_obj.release()
            elif self.movie_format == 'hdf5':
                self.writer_obj.close()
            elif self.movie_format == 'ffmpeg':
                self.writer_obj.stdin.close()
                if self.writer_obj.stderr is not None:
                    self.writer_obj.stderr.close()
                self.writer_obj.wait()
                del(self.writer_obj)

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
    def __init__(self,serial, config,
                 start_t=None,height=None,width=None,save=False,savedir=None,experiment=None, name=None,
        movie_format='hdf5',metadata_format='hdf5', preview=False,verbose=False,options=None):
        super().__init__(start_t,height, width, save, savedir, experiment, name, 
                        movie_format, metadata_format, preview, verbose, options)
        self.serial = serial
        self.config = config
        
    def process(self, left, right):
        # t0 = time.perf_counter()
        out = np.hstack((left, right))
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        # print('process t: %.6f' %( (time.perf_counter() - t0)*1000 ))
        return(out)
    
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
        self.started= True
        
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
            ir_sensors.set_option(rs.option.exposure,1200)
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