from builtin_interfaces.msg import Time

def ts2sec(stamp):
    seconds = stamp.sec + (stamp.nanosec * pow(10, -9))

    return seconds