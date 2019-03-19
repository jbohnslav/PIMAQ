import h5py
import os
import numpy as np
import argparse


datadir = r'D:\DATA\JB\realsense\testing_preview_dontsaveimages_192919_132900'
fps = 90

def main():
    parser = argparse.ArgumentParser(description='Print the dropped frame percentage from a directory.')
    parser.add_argument('-d','--directory', type=str, default=datadir,
        help='Directory with h5 files.')
    args = parser.parse_args()
    
    files = os.listdir(args.directory)
    files.sort()
    files = [os.path.join(args.directory, i) for i in files if i.endswith('.h5')]

    for file in files:
        with h5py.File(file, 'r') as f:
            framecounts = f['framecount'][:]
            # if a frame was dropped, the difference between frame counts on consecutive
            # frames will be 2 or more
            dropped_p = (np.diff(framecounts)>1.5).sum()/(framecounts.size-1)
            print('%s: %.6f' %(os.path.basename(file).ljust(14), 
                             dropped_p))

if __name__=='__main__':
    main()