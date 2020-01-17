import cv2
import h5py
import numpy as np
import subprocess as sp
from queue import Queue, Empty
from threading import Thread
from typing import Union
import os

def initialize_hdf5(filename, framesize=None, codec=None, fps=None):
    base, ext = os.path.splitext(filename)
    filename = base + '.h5'
    f = h5py.File(filename, 'w')
    datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset('frame', (0,), maxshape=(None,),dtype=datatype)
    # dset = f.create_dataset('right', (0,), maxshape=(None,),dtype=datatype)
    return(f)

def write_frame_hdf5(writer_obj, frame, axis=0, quality:int=80):
    # ret1, left_jpg = cv2.imencode('.jpg', left, (cv2.IMWRITE_JPEG_QUALITY,80))
    # ret2, right_jpg = cv2.imencode('.jpg', right, (cv2.IMWRITE_JPEG_QUALITY,80))
    ret, jpg = cv2.imencode('.jpg', frame, (cv2.IMWRITE_JPEG_QUALITY,quality))
    writer_obj['frame'].resize(writer_obj['frame'].shape[axis]+1, axis=axis)
    # f['left'].resize(f['left'].shape[axis]+1, axis=axis)
    writer_obj['frame'][-1]=jpg.squeeze()
     
def initialize_opencv(filename, framesize, codec, fps:float=30.0):
    if codec == 0:
        filename = filename + '_%06d.bmp'
        fourcc = 0
        fps=0
    else:
        # filename = filename + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*codec)
    # fourcc = -1
    writer = cv2.VideoWriter(filename,fourcc, fps, framesize)
    return(writer)

def write_frame_opencv(writer_obj, frame):
    # out = cv2.cvtColor(np.hstack((left, right)), cv2.COLOR_GRAY2RGB)
    # t0 = time.perf_counter()
    writer_obj.write(frame)
    # print('image writing t: %.6f' %( (time.perf_counter() - t0)*1000 ))
    
def initialize_ffmpeg(filename,framesize, codec=None, fps:float=30.0):
    # filename = filename + '.avi'
    size_string = '%dx%d' %framesize
    # outname = os.path.join(outdir, fname)
    fps = str(fps)
    command = [ 'ffmpeg',
        '-threads', '1',
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', size_string, # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', fps, # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'libx264',
        '-crf', '17', 
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

class DirectoryWriter:
    def __init__(self, directory, filetype, fnum:int=0):
        if os.path.isdir(directory):
            raise ValueError('Directory already exists: {}'.format(directory))
        os.makedirs(directory)
        self.directory = directory
        self.filetype = filetype
        self.fnum = fnum
        
    def write(self, frame):
        filename = os.path.join(self.directory, '{:09d}{}'.format(self.fnum,self.filetype))
        cv2.imwrite(filename, frame)
        self.fnum += 1
    
def initialize_directory(directory, framesize=None, codec=None, fps=None):
    writer_obj = DirectoryWriter(directory, filetype=codec)
    return(writer_obj)

def write_frame_directory(writer_obj, frame):
    writer_obj.write(frame)
    
class VideoWriter:
    """Class for writing videos using OpenCV, FFMPEG libx264, or HDF5 arrays of JPG bytestrings.

    OpenCV: can use encode using MJPG, XVID / DIVX, uncompressed bitmaps, or FFV1 (lossless) encoding
    FFMPEG: can use many codecs, but here only libx264, a common encoder with very high compression rates
    HDF5: Encodes each image as a jpg, and stores as an array of these jpg encoded bytestrings
        Very similar filesize to MJPG encoding, but dramatically faster RANDOM reads!
        Good for if you need often to grab a random frame from anywhere within a video, but slightly slower for
        reading sequential frames.
    directory: encodes each image as a .jpg, .png, .tiff, .bmp, etc. Saves with filename starting at 000000000.jpg

    Useful features:
        - allows for use of a context manager, so you'll never forget to close the writer object
        - Don't need to specify frame size before starting writing
        - Handles OpenCV's bizarre desire to save videos in the BGR colorspace

    Example:
        with VideoWriter('../movie.avi', movie_format = 'opencv') as writer:
            for frame in frames:
                writer.write(frame)
    """
    def __init__(self, filename: Union[str, bytes, os.PathLike], height: int = None, width: int = None,
                 fps: int = 30, movie_format: str = 'opencv', codec: str = 'MJPG', filetype='.jpg',
                 colorspace: str = 'RGB', asynchronous: bool = True, verbose: bool = False) -> None:
        """Initializes a VideoWriter object.

        Args:
            filename: name of movie to be written
            height: height (rows) in frames of movie. None: figure it out when the first frame is written
            width: width (columns) in frames of movie. None: figure it out when the first frame is written
            fps: frames per second. Does nothing for HDF5 encoding
            movie_format: one of 'opencv', 'ffmpeg', or 'hdf5'. See the class docstring for more information
            codec: encoder for OpenCV video writing. I recommend MJPG, 0, DIVX, XVID, or FFV1.
                More info here: http://www.fourcc.org/codecs.php
            filetype: the type of image to save if saving as a directory of images. 
                [.bmp, jpg, .png, .tiff]
            colorspace: colorspace of input frames. Necessary because OpenCV expects BGR. Default: RGB
            asynchronous: if True, writes in a background thread. Useful if writing to disk is slower than the image
                generation process.
            verbose: True will generate lots of print statements for debugging

        Returns:
            VideoWriter object
        """
        assert (movie_format in ['opencv', 'hdf5', 'ffmpeg', 'directory'])
        self.filename = filename
        if movie_format=='directory':
            assert(filetype in ['.bmp', '.jpg', '.png', '.jpeg', '.tiff', '.tif'])
            # save it as "codec" so that initialization and write funcs have this info
            self.codec = filetype
        else:
            base, ext = os.path.splitext(self.filename)
            if movie_format == 'opencv' or movie_format == 'ffmpeg':
                assert (ext in ['.avi', '.mp4'])
            self.codec = codec
        self.height = height
        self.width = width
        
        self.movie_format = movie_format
        if self.movie_format == 'ffmpeg':
            print('Using libx264 to encode video, ignoring codec argument...')
        self.fps = fps
        
        self.colorspace = colorspace
        assert (self.colorspace in ['BGR', 'RGB', 'GRAY'])
        self.verbose = verbose
        self.asynchronous = asynchronous

        if movie_format == 'hdf5':
            self.initialization_func = initialize_hdf5
            self.write_function = write_frame_hdf5
        elif movie_format == 'opencv':
            self.initialization_func = initialize_opencv
            self.write_function = write_frame_opencv
        elif movie_format == 'ffmpeg':
            self.initialization_func = initialize_ffmpeg
            self.write_function = write_frame_ffmpeg
        elif movie_format == 'directory':
            self.initialization_func = initialize_directory
            self.write_function = write_frame_directory
        framesize = (self.width, self.height)

        if self.asynchronous:
            self.save_queue = Queue(maxsize=3000)
            self.save_thread = Thread(target=self.save_worker, args=(self.save_queue,))
            self.save_thread.daemon = True
            self.save_thread.start()
        self.has_stopped = False
        self.writer_obj = None

    def save_worker(self, queue):
        """Worker for asychronously writing video to disk"""
        should_continue = True
        while should_continue:
            try:
                item = queue.get()
                if item is None:
                    if self.verbose:
                        print('Saver stop signal received')
                    should_continue = False
                    break
                self.write_frame(item)
                # print(queue.qsize())
            except Exception as e:
                print(e)
            finally:
                queue.task_done()
        if self.verbose:
            print('out of save queue')

    def write(self, frame: np.ndarray):
        """Writes numpy array to disk"""
        if self.asynchronous:
            self.save_queue.put(frame)
        else:
            self.write_frame(frame)

    def write_frame(self, frame: np.ndarray):
        """Writes numpy array to disk. Doesn't happen in background thread: for that use `write`

        Args:
            frame: numpy ndarray of shape (H,W,C) or (H,W). channel will be added in the case of a 2D array
        """
        # get shape
        if frame.ndim == 3:
            H, W, C = frame.shape
        # add a gray channel if necessary
        elif frame.ndim == 2:
            H, W = frame.shape
            C = 1
            frame = frame[..., np.newaxis]
        else:
            raise ValueError('Unknown frame dimensions: {}'.format(frame.shape))

        # use the first frame to get height and width
        if self.height is None:
            self.height = H
        if self.width is None:
            self.width = W

        # initialize the writer object. Could be OpenCV VideoWriter, subprocessing Pipe, or HDF5 File
        if self.writer_obj is None:
            self.writer_obj = self.initialization_func(self.filename,
                                                       (self.width, self.height), self.codec, self.fps)

        if frame.dtype == np.uint8:
            pass
        elif frame.dtype == np.float:
            # make sure that frames are in proper format before writing. We don't want the writer to be implicitly
            # changing pixel values, that should be done outside of this Writer class
            assert (frame.min() >= 0 and frame.max() <= 1)
            frame = (frame * 255).clip(min=0, max=255).astype(np.uint8)
        # opencv expects BGR format
        if self.colorspace == 'BGR':
            if self.movie_format != 'opencv':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.colorspace == 'RGB':
            if self.movie_format == 'opencv':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.colorspace == 'GRAY':
            if self.movie_format == 'opencv':
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # actually write to disk
        self.write_function(self.writer_obj, frame)

    def __enter__(self):
        # allows use with decorator
        return self

    def __exit__(self, type, value, traceback):
        # allows use with decorator
        self.stop()

    def stop(self):
        """Stops writing, closes all open file objects"""
        if self.has_stopped:
            return
        if self.asynchronous:
            # wait for save worker to complete, then finish
            self.save_queue.put(None)
            if self.verbose:
                print('joining...')
            self.save_queue.join()
            if self.verbose:
                print('joined.')
            del (self.save_queue)
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
                del (self.writer_obj)
        self.has_stopped = True

    def __del__(self):
        """Destructor"""
        try:
            self.stop()
        except BaseException as e:
            if self.verbose:
                print('Error in destructor')
                print(e)
            else:
                pass