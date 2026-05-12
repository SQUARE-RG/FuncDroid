from .proto import OperatingSystem


def get_android_available_devices():
    """
    Get a list of device serials connected via adb
    :return: list of str, each str is a device serial number
    """
    import subprocess
    r = subprocess.check_output(["adb", "devices"])
    if not isinstance(r, str):
        r = r.decode()
    devices = []
    for line in r.splitlines():
        segs = line.strip().split()
        if len(segs) == 2 and segs[1] == "device":
            devices.append(segs[0])
    return devices

def get_harmony_available_devices():
    """
    Get a list of device serials connected via adb
    :return: list of str, each str is a device serial number
    """
    import subprocess
    r = subprocess.check_output(["hdc", "list", "targets"])
    if not isinstance(r, str):
        r = r.decode()
    devices = []
    if '[Empty]' in r:
        return devices
    for line in r.splitlines():
        segs = line.strip().split()
        devices.append(segs[0])
    return devices