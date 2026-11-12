r"""
AeroEye-3D — radar sound effects. Sonar ping, lock tone, boot ticks.

The sounds are synthesised here rather than shipped as files, so there is
nothing to lose and no dependency beyond numpy. They are written once to a temp
folder because winsound can only play asynchronously from a file on disk.

Playback runs on its own thread: PlaySound can take milliseconds to open the
audio device, and the tracking loop must never wait for a sound effect. Any
audio failure simply means silence; it never reaches the tracker.

    import radar_sound
    s = radar_sound.Sound()
    s.play("lock")
"""
import os
import queue
import tempfile
import threading
import time
import wave

import numpy as np

try:
    import winsound
except ImportError:                 # not on Windows: silent, not broken
    winsound = None

RATE = 22050
FOLDER = os.path.join(tempfile.gettempdir(), "aeroeye_radar_sounds")
HIGH = ("lock", "lost", "ready")    # these may not be cut off by a ping


def _tone(freq, dur, decay=6.0, vol=0.6):
    t = np.arange(int(RATE * dur)) / RATE
    y = vol * np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)
    fade = min(len(y), int(RATE * 0.004))       # no click at either end
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


def _mix(*parts):
    """(start_seconds, samples)... mixed into one buffer."""
    n = max(int(RATE * s) + len(y) for s, y in parts)
    out = np.zeros(n)
    for s, y in parts:
        i = int(RATE * s)
        out[i:i + len(y)] += y
    return out


def _ping():
    body = _tone(1250, 0.9, 5.0, 0.42) + _tone(2500, 0.9, 9.0, 0.10)
    return _mix((0, body), (0.16, body * 0.32), (0.32, body * 0.12))


def _scan():
    return _tone(720, 0.35, 11.0, 0.14)


def _lock():
    beeps = [(i * 0.11, _tone(f, 0.10, 2.0, 0.45)) for i, f in enumerate((988, 1319, 1760))]
    return _mix(*beeps, (0.33, _tone(1760, 0.32, 5.0, 0.40)))


def _lost():
    return _mix((0, _tone(880, 0.16, 4.0, 0.36)), (0.17, _tone(587, 0.26, 4.0, 0.36)))


def _tick():
    return _tone(1800, 0.03, 30.0, 0.30)


def _ready():
    notes = [(i * 0.10, _tone(f, 0.55, 4.0, 0.26)) for i, f in enumerate((523, 659, 784, 1047))]
    return _mix(*notes)


SOUNDS = {"ping": _ping, "scan": _scan, "lock": _lock, "lost": _lost,
          "tick": _tick, "ready": _ready}


def _write(path, y):
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


class Sound:
    def __init__(self, enabled=True):
        self.on = bool(enabled and winsound is not None)
        self.muted = False
        self.paths, self.secs = {}, {}
        self.busy_until = 0.0
        self.q = queue.Queue(maxsize=2)
        if not self.on:
            return
        try:
            os.makedirs(FOLDER, exist_ok=True)
            for name, make in SOUNDS.items():
                y = make()
                self.paths[name] = os.path.join(FOLDER, name + ".wav")
                self.secs[name] = len(y) / RATE
                _write(self.paths[name], y)
        except OSError:
            self.on = False
            return
        threading.Thread(target=self._run, daemon=True).start()

    def play(self, name):
        """Never blocks. A ping is dropped rather than cutting off a lock tone."""
        if not self.on or self.muted:
            return
        now = time.time()
        if name in HIGH:
            self.busy_until = now + self.secs[name]
        elif now < self.busy_until:
            return
        try:
            self.q.put_nowait(name)
        except queue.Full:
            pass

    def _run(self):
        flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
        while True:
            name = self.q.get()
            try:
                winsound.PlaySound(self.paths[name], flags)
            except RuntimeError:
                pass

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted and self.on:
            winsound.PlaySound(None, winsound.SND_PURGE)
        return self.muted
