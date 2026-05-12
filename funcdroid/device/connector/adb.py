from .connector import Connector
from hmbot.utils.exception import DeviceError, ADBError
from hmbot.utils.proto import PageInfo, Resource, AudioInfo, AudioType, Status, CameraInfo, CameraType
from loguru import logger
import subprocess
import re

try:
    from shlex import quote  # Python 3
except ImportError:
    from pipes import quote  # Python 2


class ADB(Connector):
    def __init__(self, device=None):
        from hmbot.device.device import Device
        if isinstance(device, Device):
            self.serial = device.serial
        else:
            raise DeviceError
        self.cmd_prefix = ['adb', "-s", device.serial]
        self.info = self.page_info()

    def run_cmd(self, extra_args):
        if isinstance(extra_args, str):
            extra_args = extra_args.split()
        if not isinstance(extra_args, list):
            msg = "invalid arguments: %s\nshould be str, %s given" % (extra_args, type(extra_args))
            logger.warning(msg)
            raise ADBError(msg)

        args = [] + self.cmd_prefix
        args += extra_args

        # logger.debug('command:')
        # logger.debug(args)
        r = subprocess.check_output(args).strip()
        if not isinstance(r, str):
            r = r.decode()
        # logger.debug('return:')
        # logger.debug(r)
        return r

    def shell(self, extra_args):
        if not isinstance(extra_args, str):
            msg = "invalid arguments: %s\nshould be str, %s given" % (extra_args, type(extra_args))
            logger.warning(msg)
            raise ADBError(msg)
        extra_args = 'shell ' + extra_args
        return self.run_cmd(extra_args)

    def shell_grep(self, extra_args, grep_args):
        if isinstance(extra_args, str):
            extra_args = extra_args.split()
        if isinstance(grep_args, str):
            grep_args = grep_args.split()
        if not isinstance(extra_args, list) or not isinstance(grep_args, list):
            msg = "invalid arguments: %s\nshould be str, %s given" % (extra_args, type(extra_args))
            logger.warning(msg)
            raise ADBError(msg)

        args = self.cmd_prefix + ['shell'] + [quote(arg) for arg in extra_args]
        grep_args = ['grep'] + [quote(arg) for arg in grep_args]

        proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
        proc2 = subprocess.Popen(grep_args, stdin=proc1.stdout,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        proc1.stdout.close()  # Allow proc1 to receive a SIGPIPE if proc2 exits.
        out, err = proc2.communicate()
        if not isinstance(out, str):
            out = out.decode()
        return out

    def grep(self, shell_out, grep_args):
        shell_out = shell_out.splitlines()
        return [s for s in shell_out if grep_args in s]

    def page_info(self):
        focus_lines = self.shell_grep("dumpsys window", "mCurrentFocus").splitlines()
        infos_re = re.compile(".*u0 (.*)/(.*)}")
        if len(focus_lines) > 0:
            for focus_line in focus_lines:
                focus_line_str = focus_line.decode('utf-8') if isinstance(focus_line, bytes) else focus_line
                match = infos_re.match(focus_line_str)
                if match:
                    self.info = PageInfo(bundle=match.groups()[0],
                                        ability=match.groups()[1],
                                        name=match.groups()[1])
                    return self.info
                else:
                    self.info = PageInfo(bundle='PopupWindow', ability='PopupWindow', name='PopupWindow')
                    return self.info
        return None

    def get_uid(self, bundle=None):
        if not bundle:
            if not self.info:
                self.info = self.page_info()
            bundle = self.info.bundle
        process_lines = self.shell_grep("ps", bundle).splitlines()
        if len(process_lines) > 0:
            usr_name = process_lines[0].split()[0]
            uid = str(int(usr_name.split('_a')[1]) + 10000)
            return uid
        else:
            return

    def get_resources(self, bundle=None):
        if not bundle:
            if not self.info:
                self.info = self.page_info()
            if not self.info:
                return Resource(None, None)
            bundle = self.info.bundle
        return Resource(audio=self.get_audio(bundle),
                        camera=self.get_camera(bundle))

    def get_audio(self, bundle=None):
        return AudioInfo(type=AudioType.MUSIC, stat=Status.STOPPED)
        if not bundle:
            if not self.info:
                self.info = self.page_info()
            bundle = self.info.bundle
        # print('bundle=', bundle)
        all_audio_lines = self.shell("dumpsys audio")
        # todo
        type = AudioType.MUSIC

        audio_lines = self.grep(all_audio_lines, "AudioPlaybackConfiguration")
        audio_line_re = re.compile(".*u/pid:(.*)/(.*) .*state:(.*) attr.*")
        audio_status_dict = {}
        for audio_line in audio_lines:
            m = audio_line_re.match(audio_line)
            if m:
                uid = m.group(1)
                pid = m.group(2)
                status = m.group(3)
                if (uid, pid) not in audio_status_dict:
                    audio_status_dict[(uid, pid)] = status
                elif status == 'started':
                    audio_status_dict[(uid, pid)] = status
        # print(f'audio_status_dict={audio_status_dict}')

        req_focus_lines = self.grep(all_audio_lines, "requestAudioFocus")
        req_focus_line_re = re.compile(".*uid/pid (\d*)/(\d*) .*clientId=(.*) callingPack=.*")
        client_dict = {}
        for req_focus_line in req_focus_lines:
            m = req_focus_line_re.match(req_focus_line)
            if m:
                uid = str(m.group(1))
                pid = str(m.group(2))
                client_id = m.group(3)
                client_dict[client_id] = (uid, pid)

        focus_lines = self.grep(all_audio_lines, "source")
        focus_line_re = re.compile(".* pack: (.*) -- client: (.*) -- gain: (.*) -- flags.* loss: (.*) -- notified.*")
        focus_dict = {}
        for focus_line in focus_lines:
            m = focus_line_re.match(focus_line)
            if m:
                (uid, pid) = client_dict[m.group(2)]
                focus_dict[(uid, pid)] = (m.group(3), m.group(4))
        # print(f'focus_dict={focus_dict}')

        uid_ = self.get_uid(bundle)
        # print(f'uid={uid_}')
        stat = Status.STOPPED
        for (uid, pid), status in audio_status_dict.items():
            if uid != uid_:
                continue
            # print(uid, pid, status)
            if status == 'started':
                if not focus_dict:
                    return AudioInfo(type, Status.STOPPED)
                return AudioInfo(type, Status.RUNNING)
        return AudioInfo(type, stat)

    def get_camera(self, bundle=None):
        # todo
        return CameraInfo(type=CameraType.FRONT,
                         stat=Status.STOPPED)

