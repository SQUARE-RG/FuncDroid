from abc import ABC, abstractmethod
import re
from ..utils.proto import SystemKey

class Event(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def _json(self):
        pass

class ClickEvent(Event):
    def __init__(self, node):
        self.node = node

    def execute(self):
        self.node.click()

    def _json(self):
        return {
            'type': 'Click',
            'node': self.node._json()
        }

class LongClickEvent(Event):
    def __init__(self, node):
        self.node = node

    def execute(self):
        self.node.long_click()

    def _json(self):
        return {
            'type': 'LongClick',
            'node': self.node._json()
        }

class InputEvent(Event):
    def __init__(self, node, text):
        self.node = node
        self.text = text

    def execute(self):
        self.node.input(self.text)

    def _json(self):
        return {
            'type': 'Input',
            'text': self.text,
            'node': self.node._json(),
        }

class SwipeExtEvent(Event):
    def __init__(self, device, window, direction):
        self.device = device
        self.window = window
        self.direction = direction

    def execute(self):
        if self.direction == 'down':
            self.device.swipe_ext('up')
        elif self.direction == 'up':
            self.device.swipe_ext('down')
        elif self.direction == 'left':
            self.device.swipe_ext('left')
        elif self.direction == 'right':
            self.device.swipe_ext('right')

    def _json(self):
        return {
            'type': 'SwipeExt',
            'direction': self.direction,
        }

class KeyEvent(Event):
    def __init__(self, device, window, key):
        self.device = device
        self.window = window
        self.key = key

    def execute(self):
        getattr(self.device, self.key)()

    def _json(self):
        return {
            'type': 'Key',
            'key': self.key,
        }

class StartAppEvent(Event):
    def __init__(self, device, app):
        self.device = device
        self.app = app

    def execute(self):
        self.device.start_app(self.app)

    def _json(self):
        return {
            'type': 'StartApp',
            'app': self.app,
        }