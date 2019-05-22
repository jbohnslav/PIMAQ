# PIMAQ
## Python Image Acquisition 
### A software package for simultaneous video acquisition from multiple cameras. 
Motivation: Using multiple cameras for measuring animal behavior is very common. For single camera acquisition, the hardware manufacturer (e.g. PointGrey, Basler, etc) wrote their own software with GUIs for changing camera attributes and acquiring data simply. MATLAB Image Acquisition toolbox also has its own GUI for previewing and saving videos. However, these GUIs do not easily support multiple simultaneous camera acquisition. Furthermore, manufacturers often make C++ or Python APIs for controlling the cameras, but in my experience they are poorly documented and hard to use. I wrote this package to make it simple and easy to acquire data from many cameras simultaneously.

### Currently supported cameras
* Intel Realsense (tested with D435)
* FLIR (formerly Point Grey. Tested with Flea3, FL3-U3-13Y3M-c)

### Features
* Supports both viewing video streams and saving them to disk
* Write camera parameters into simple `.yaml` configuration files
* Uses both multiprocessing and threading. 
  * Each camera loop runs in one process, meaning it uses a dedicated CPU core. 
  * The acquisition loop runs in the main thread. Previewing and Saving occurs in separate threads using Queues so that frames are not dropped because the acquisition loop waits for file saving.

### Hardware recommendations
* To save disk space, PIMAQ compresses video (with either OpenCV or ffmpeg) on-the-fly. Therefore, I recommended having a computer with at least 1 core per camera, and potentially +1 more for acquiring hardware signals (I use [Janelia's WaveSurfer](https://wavesurfer.janelia.org/))
* Storage: solid state drives are essential so that writing to disk is not a bottleneck. If it is, the saving queues will fill and acquisition will stop. With `MJPG` encoding and `4x 640x480` RealSenses and `1x 640x512` PointGrey, I'm saving about ~20MB/s. Random write speeds with a modern SSD is about ~60MB/s, so this is plenty. However, if you have tons of cameras, you will need either a RAID array of SSDs or one NVMe (not M2!) drive.

# Installation
### Installing PIMAQ
Dependencies:
```
numpy
OpenCV
h5py
optional: ffmpeg (for libx264 encoded videos)
optional: PySpin (for FLIR cameras)
optional: pyrealsense2 (for realsense cameras)
```
Installation:
``` 
git clone https://github.com/jbohnslav/panopticon.git
```
That's it! 

### Installing pyrealsense2 to control realsense cameras
For full installation instructions, [go to Intel's Github page](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation)
Short version: `pip install pyrealsense2`

### installing PySpin to control FLIR (formerly PointGrey) cameras
* Go to the Spinnaker downloads page at FLIR (formerly PointGrey) [here.](https://www.flir.com/products/spinnaker-sdk/)
* Click Download Now. Note: I had to disable AdBlock, etc. to get the website to show the download button.
* Click your OS (e.g. Windows)
* Click Latest Spinnaker Web Installer. Download the x64 version and install. Note the version!!
  * You might be able to get away with installing only "Visual Studio runtimes and drivers."
* Go back one page and click on the "Latest Python Spinnaker"
  * Note: a full version of these instructions can be found in the README in this zip file!
  * Download the .zip file corresponding to your python installation and OS version. E.g., my python version is 3.6 and my OS is 64 bit, so I downloaded `spinnaker_python-1.20.0.15-cp36-cp36m-win_amd64`
    * The `cp36` means python 3.6, and `amd64` means 64 bit.
* Unzip this file
* `cd` into this file location. There should be a file like `spinnaker_python-1.23.0.27-cp36-cp36m-win_amd64.whl`
* activate your anaconda or pip environment!!!
* `python -m ensurepip`: ensures that pip is installed
* `python -m pip install --upgrade pip numpy`. This makes sure that `numpy` is installed
* `python -m pip install spinnaker_python-1.x.x.x-cp36-cp36m-win_amd64.whl` to install PySpin, with of course the correct filename for your version.
* `python Examples\Python3\Acquisition.py` Running this example will verify your installation, as long as you have a FLIR camera connected. 
* Potential bugs
  * It's very important to run the `Acquistion.py` example to catch any potential bugs.
  * I encountered a strange bug relating to my `anaconda` numpy version. It said something about `MKL_DNN` or something.
    * If this happens to you, run `python -m pip install --upgrade --force-reinstall spinnaker_python-1.20.0.15-cp36-cp36m-win_amd64.whl` (with the correct version). This will reinstall the correct numpy version.
