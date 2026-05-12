from enum import Enum
from dataclasses import dataclass

class OperatingSystem(str, Enum):
    HARMONY = 'harmony'
    ANDROID = 'android'

class SwipeDirection(str, Enum):
    LEFT = 'left'
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

class SystemKey(str, Enum):
    BACK = 'back'
    HOME = "home"
    RECENT = "recent"
    HOP = "hop"
    RESTART = "restart_app"

class ExploreStrategy(str, Enum):
    DFS = 'dfs'
    BFS = 'bfs'
    QLN = 'qln'
    LLM = 'llm'

class ExploreMission(str, Enum):
    IMC = 'INITIAL_MODEL_CONSTRUCTION'

class TerminateCondition(str, Enum):
    IMC = 'INITIAL_MODEL_CONSTRUCTION'

class LLMUrl(str, Enum):
    DS = 'https://api.deepseek.com'

class DisplayRotation(int, Enum):
    ROTATION_0 = 0
    ROTATION_90 = 1
    ROTATION_180 = 2
    ROTATION_270 = 3

@dataclass
class DisplayInfo:
    sdk: str
    width: int
    height: int
    rotation: DisplayRotation

class ExploreGoal(str, Enum):
    TESTCASE = 'testcase'
    HARDWARE = 'hardware'

class AudioType(str, Enum):
    MUSIC = 'music'
    MOVIE = 'movie'
    NAVIG = 'navigation'
    COMMU = 'communication'

class CameraType(str, Enum):
    FRONT = 'front'
    REAR = 'rear'

class Status(str, Enum):
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'

@dataclass
class AudioInfo: 
    type: AudioType
    stat: Status

@dataclass
class CameraInfo: 
    type: CameraType
    stat: Status

@dataclass
class Resource:
    audio: AudioInfo
    camera: CameraInfo

@dataclass
class PageInfo:
    bundle: str
    ability: str
    name: str
