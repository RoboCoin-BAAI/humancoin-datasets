from queue import Full, Queue
from threading import Event, Thread
from time import perf_counter

import av

from .video import VideoWriter


def prefetch_rgb(input_video, count, depth, stats):
    queue = Queue(maxsize=depth)
    stopped = Event()

    def send(value):
        while not stopped.is_set():
            try:
                queue.put(value, timeout=0.1)
                return
            except Full:
                pass

    def decode():
        try:
            context = av.open(str(input_video))
            with context as container:
                frames = container.decode(video=0)
                iterator = iter(frames)
                for _ in range(count):
                    if stopped.is_set():
                        break
                    start = perf_counter()
                    try:
                        frame = next(iterator)
                    except StopIteration:
                        break
                    rgb = frame.to_ndarray(format="rgb24")
                    stats["decode_rgb_seconds"] += perf_counter() - start
                    send((rgb, None))
            send((None, None))
        except BaseException as error:
            send((None, error))

    thread = Thread(target=decode, name="overlay-decode", daemon=True)
    thread.start()
    try:
        while True:
            rgb, error = queue.get()
            if error is not None:
                raise error
            if rgb is None:
                break
            yield rgb
    finally:
        stopped.set()
        thread.join()


class AsyncVideoWriter:
    def __init__(self, path, width, height, fps, *, codec, depth):
        self.queue = Queue(maxsize=depth)
        self.error = None
        self.ready = Event()
        self.closed = False
        self.stats = {"encode_seconds": 0.0, "encode_queue_wait_seconds": 0.0}

        def encode():
            writer = None
            try:
                writer = VideoWriter(path, width, height, fps, codec=codec, faststart=True)
                self.codec, self.options = (writer.codec, writer.options)
                self.fallback_reason = writer.fallback_reason
                self.ready.set()
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    rgb, timestamp = item
                    start = perf_counter()
                    writer.write(rgb, timestamp)
                    self.stats["encode_seconds"] += perf_counter() - start
            except BaseException as error:
                self.error = error
            finally:
                if writer is not None:
                    start = perf_counter()
                    try:
                        writer.close()
                    except BaseException as error:
                        if self.error is None:
                            self.error = error
                    self.stats["encode_seconds"] += perf_counter() - start
                self.ready.set()

        self.thread = Thread(target=encode, name="overlay-encode", daemon=True)
        self.thread.start()
        self.ready.wait()
        self._check_error()

    def _check_error(self):
        if self.error is not None:
            raise self.error

    def _put(self, item):
        while True:
            self._check_error()
            try:
                self.queue.put(item, timeout=0.1)
                break
            except Full:
                pass
        self._check_error()

    def write(self, rgb, timestamp):
        if self.closed:
            raise ValueError("Overlay writer is closed")
        start = perf_counter()
        self._put((rgb, timestamp))
        self.stats["encode_queue_wait_seconds"] += perf_counter() - start

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._put(None)
            finally:
                self.thread.join()
        self._check_error()
