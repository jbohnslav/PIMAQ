import pyrealsense2 as rs

def enumerate_connected_devices():
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
    context = rs.context()
    connect_device = []
    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))
    return connect_device