import signal
import threading
from types import FrameType
from typing import Callable, Self
from time import sleep
from abc import ABC, abstractmethod


class Alarm(ABC):
    @abstractmethod
    def __init__(self: Self, func: Callable[[], None], time: float) -> None:
        pass
    
    @abstractmethod
    def start(self: Self) -> None:
        pass

    @abstractmethod
    def stop(self: Self) -> None:
        pass


class SignalAlarm(Alarm):
    def __init__(self: Self, func: Callable[[], None], time: float) -> None:
        self.func = func
        self.time = time


    def start(self: Self) -> None:
        def wrapped(sig: int, frame: FrameType|None):
            self.func()
            signal.setitimer(signal.ITIMER_REAL, self.time)

        signal.signal(signal.SIGALRM, wrapped)
        signal.setitimer(signal.ITIMER_REAL, self.time)


    def stop(self: Self) -> None:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL, 0)


class ThreadAlarm(Alarm):
    def __init__(self: Self, func: Callable[[], None], time: float) -> None:
        self.func = func
        self.time = time
        self.stop_event = threading.Event()
        self.thread: threading.Thread|None = None

    
    def start(self: Self) -> None:
        def thread(stop_event: threading.Event):
            while not stop_event.is_set():
                self.func()
                sleep(self.time)

        self.thread = threading.Thread(target=thread, args=(self.stop_event,))
        self.thread.start()


    def stop(self: Self) -> None:
        assert isinstance(self.thread, threading.Thread), "Attempted to stop a timer that has not been started"
        
        self.stop_event.set()
        self.thread.join()
