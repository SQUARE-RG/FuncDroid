from abc import ABC, abstractmethod
class App(ABC):
    """
    this interface describes a App (Android or Harmony)
    """

    @abstractmethod
    def __init__(self, app_path='', device=None):
        pass
