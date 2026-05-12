from hmbot.device.connector.adb import ADB
from hmbot.device.connector.hdc import HDC
from hmbot.device.automator.u2 import U2
from hmbot.device.automator.h2 import H2
from ..proto import OperatingSystem

system_rfl = {
    OperatingSystem.ANDROID: (ADB, U2),
    OperatingSystem.HARMONY: (HDC, H2)
}