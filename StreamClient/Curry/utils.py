import struct

def typecast_uint16_from_bytes(b):
    return [int.from_bytes(b[2 * i:2 * i + 1], byteorder='big') for i in range(int(len(b) / 2))]

def typecast_int16_from_bytes(b):
    return [int.from_bytes(b[2 * i:2 * i + 1], byteorder='big', signed=True) for i in range(int(len(b) / 2))]

def typecast_double_from_bytes(b):
    return struct.unpack('d',b)

def typecast_float_from_bytes(b):
    return struct.unpack('f',b)

def typecast_single_from_bytes(b):
    return [struct.unpack('f',b[4*i:4*i+4]) for i in range(int(len(b)/4))]

def typecast_uint32_from_bytes(b):
    return struct.unpack('q',b)



