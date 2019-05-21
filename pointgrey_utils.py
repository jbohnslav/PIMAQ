import PySpin 

def get_serial_number(cam):
    return(PySpin.CValuePtr(
        cam.GetTLDeviceNodeMap().GetNode('DeviceSerialNumber')).ToString()
    )

def set_value(nodemap, nodename, value):
    try:
        node = nodemap.GetNode(nodename)
        nodetype = node.GetPrincipalInterfaceType()

        nodeval, typestring = get_nodeval_and_type(node)

        assert(PySpin.IsWritable(nodeval))

        if typestring == 'int' or typestring == 'float':
            assert(value <= nodeval.GetMax() and value >= nodeval.GetMin())
        if typestring == 'int':
            assert(type(value)==int)
            if PySpin.IsAvailable(nodeval) and PySpin.IsWritable(nodeval):
                nodeval.SetValue(value)
            else:
                raise ValueError('Node not writable or available: %s' %nodename)

        elif typestring == 'float':
            assert(type(value)==float)
            if PySpin.IsAvailable(nodeval) and PySpin.IsWritable(nodeval):
                nodeval.SetValue(value)
            else:
                raise ValueError('Node not writable or available: %s' %nodename)
        elif typestring == 'enum':
            assert(type(value)==str)

            entry = nodeval.GetEntryByName(value)

            if entry is None:
                print('Valid entries: ')
                entrylist = nodeval.GetEntries()
                for entry in entrylist:
                    print(entry.GetName())
                raise ValueError('Invalid entry!: %s' %value)
            else:
                entry = PySpin.CEnumEntryPtr(entry)
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                nodeval.SetIntValue(entry.GetValue())
            else:
                raise ValueError('Entry not readable!')
            # PySpin.CEnumEntryPtr
        elif typestring == 'bool':
            assert(type(value)==bool)
            if PySpin.IsAvailable(nodeval) and PySpin.IsWritable(nodeval):
                nodeval.SetValue(value)
            else:
                raise ValueError('Node not writable or available: %s' %nodename)
    except PySpin.SpinnakerException as e:
        raise ValueError('Error: %s' %e)


def turn_strobe_on(nodemap, line, strobe_duration=0.0):
    assert(type(line)==int)
    assert(type(strobe_duration)==float)
    
    linestr = 'Line%d'%line
    
    # set the line selector to this line so that we change the following
    # values for Line2, for example, not Line0
    set_value(nodemap, 'LineSelector', linestr)
    # one of input, trigger, strobe, output
    set_value(nodemap, 'LineMode', 'Strobe')
    # enable strobe
    set_value(nodemap, 'StrobeEnabled', True)
    # set duration, in us I think?
    set_value(nodemap, 'StrobeDuration', strobe_duration)
    # inverted means low by default, high when strobe is on
    set_value(nodemap, 'LineInverter', True)

def print_value(nodemap, nodename):
    assert(type(nodename)==str)
    node = nodemap.GetNode(nodename)
    nodeval, typestring = get_nodeval_and_type(node)
    if typestring == 'enum':
        # GetCurrentEntry
        print(nodename, typestring, nodeval.ToString())
    else:
        print(nodename, typestring, nodeval.GetValue())

def get_nodeval_and_type(node):
    nodetype = node.GetPrincipalInterfaceType()
    if nodetype== PySpin.intfIString:
        nodeval = PySpin.CStringPtr(node)
        typestring = 'string'
    elif nodetype== PySpin.intfIInteger:
        nodeval = PySpin.CIntegerPtr(node)
        typestring = 'int'
    elif nodetype== PySpin.intfIFloat:
        nodeval = PySpin.CFloatPtr(node)
        typestring = 'float'
    elif nodetype== PySpin.intfIBoolean:
        nodeval = PySpin.CBooleanPtr(node)
        typestring = 'bool'
    elif nodetype == PySpin.intfIEnumeration:
        nodeval = PySpin.CEnumerationPtr(node)
        typestring = 'enum'
    elif nodetype == PySpin.intfICommand:
        nodeval = PySpin.CCommandPtr(node)
        typestring = 'command'
    else:
        raise ValueError('Invalid node type: %s' %nodetype)
        
    return(nodeval, typestring)

