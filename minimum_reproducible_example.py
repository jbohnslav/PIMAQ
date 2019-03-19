import pyrealsense2 as rs
import numpy as np
import cv2

preview = True
SERIAL = '830112071475'

def main():

    config = rs.config()
    resolution_width = 480
    resolution_height = 270
    framerate = 90
    config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_stream(rs.stream.infrared, 2, resolution_width, resolution_height, rs.format.y8, framerate)
    config.enable_device(SERIAL)

    pipe = rs.pipeline()
    prof = pipe.start(config)
    dev = prof.get_device()
    ds = dev.query_sensors()[0]
    ds.set_option(rs.option.inter_cam_sync_mode, 1)
    # ds.set_option(rs.option.frames_queue_size,1)
    print(ds.get_option(rs.option.inter_cam_sync_mode))

    framecount = 0
    if preview:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.namedWindow('realsense', cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            # print('acquiring')
            frames = pipe.wait_for_frames(1000*15)
            # frames = device.pipeline.poll_for_frames()
            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            left, right = np.asanyarray(left.get_data()), np.asanyarray(right.get_data())

            if preview:
                out = np.vstack((left,right))
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                # string = '%.4f' %(time_acq*1000)
                string = '%07d' %(framecount)
                cv2.putText(out,string,(10,500), font, 0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.imshow('realsense', out)
                key = cv2.waitKey(1)
                if key==27:
                    break
            framecount+=1
    except KeyboardInterrupt:
        pass
        # print('User stopped acquisition.')
    finally:
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()

