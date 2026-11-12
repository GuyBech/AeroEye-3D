# AeroEye-3D promo soundtrack: everything synthesised here, on the same 100 BPM grid as the picture
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 48000
DUR = 74.4
BEAT, BAR = 0.6, 2.4
N = int(DUR * SR) + SR
rng = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))


def bus():
    return np.zeros((N, 2), np.float32)


music, drums, fx, vo, wet = bus(), bus(), bus(), bus(), bus()


def put(b, sig, t, g=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N:
        return
    sig = sig[:N - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    if sig.ndim == 1:
        b[i:i + len(sig), 0] += sig * g * l * 1.41
        b[i:i + len(sig), 1] += sig * g * r * 1.41
    else:
        b[i:i + len(sig)] += sig * g


def T(d):
    return np.arange(int(d * SR)) / SR


def flt(x, kind, fc, order=2):
    return sosfilt(butter(order, fc, kind, fs=SR, output="sos"), x, axis=0)


def sweep_noise(d, f0, f1, q=.9, peak=.75, stereo=False):
    """Band-limited noise whose band follows f0 -> f1, by overlap-add of blocks."""
    n = int(d * SR)
    x = rng.normal(0, 1, n)
    hop = 1024
    win = np.hanning(hop * 2)
    out = np.zeros(n)
    for k in range(0, n - hop * 2, hop):
        u = k / n
        fc = f0 * (f1 / f0) ** u
        lo, hi = max(fc / (1 + q), 30), min(fc * (1 + q), SR / 2 - 200)
        seg = sosfilt(butter(2, (lo, hi), "band", fs=SR, output="sos"), x[k:k + hop * 2])
        out[k:k + hop * 2] += seg * win
    e = np.minimum(T(d) / (d * peak), (d - T(d)) / (d * (1 - peak) + 1e-6)).clip(0, 1) ** 1.6
    return out * e / (np.abs(out).max() + 1e-9)


def kick(g=1.0):
    t = T(.42)
    f = 46 + 95 * np.exp(-t * 26)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.5) * g + flt(rng.normal(0, 1, len(t)), "high", 2500) * np.exp(-t * 260) * .25


def taiko(g=1.0, f0=62):
    t = T(1.1)
    f = f0 + 70 * np.exp(-t * 30)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.2)
    n = flt(rng.normal(0, 1, len(t)), "low", 900) * np.exp(-t * 24) * .6
    return np.tanh((b + n) * 1.6) * g


def hat(g=1.0, d=.05):
    t = T(d * 3)
    return flt(rng.normal(0, 1, len(t)), "high", 7500) * np.exp(-t / d * 3) * g


def clap(g=1.0):
    t = T(.28)
    return flt(rng.normal(0, 1, len(t)), "band", (900, 4200)) * np.exp(-t * 19) * g


def saw(f, t, nh=26, cutoff=None):
    """Additive band-limited saw; cutoff is a per-sample array of Hz."""
    out = np.zeros_like(t)
    ph = 2 * np.pi * f * t
    for h in range(1, nh + 1):
        if f * h > SR / 2 - 500:
            break
        a = 1 / h
        if cutoff is not None:
            a = a / (1 + (f * h / cutoff) ** 4)
        out += a * np.sin(ph * h + h * .7)
    return out


def braam(f0, d=3.2, g=1.0):
    t = T(d)
    cut = 180 + 2600 * np.exp(-t * 2.4) * (1 - np.exp(-t * 26))
    x = sum(saw(f0 * r, t, 22, cut) for r in (1, 1.004, .996, 1.5, 2.0))
    sub = np.sin(2 * np.pi * f0 * .5 * t)
    e = (1 - np.exp(-t * 34)) * np.exp(-t * 1.25)
    return np.tanh((x * .5 + sub * .8) * e * 1.3) * g


def pad(freqs, d, g=1.0, cut=900):
    t = T(d)
    c = np.full_like(t, cut)
    x = sum(saw(f * s, t, 14, c) for f in freqs for s in (1, 1.003))
    e = np.clip(t / .5, 0, 1) * np.clip((d - t) / .5, 0, 1)
    return x * e * g / len(freqs)


def ping(f, g=1.0, d=1.1):
    t = T(d)
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 5.5) * (1 - np.exp(-t * 500)) * g


def beep(f, d=.07, g=1.0):
    t = T(d)
    return np.sin(2 * np.pi * f * t) * np.minimum(1, t * 800) * np.minimum(1, (d - t) * 800) * g


def tick(g=1.0):
    t = T(.012)
    return np.sin(2 * np.pi * 3100 * t) * np.exp(-t * 500) * g


def servo(d, g=1.0):
    t = T(d)
    f = 150 + 45 * np.sin(2 * np.pi * 1.3 * t) + 25 * np.sin(2 * np.pi * 7.1 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sign(np.sin(ph)) * .6 + np.sin(2.5 * ph) * .4
    x = flt(x, "band", (300, 2200))
    e = np.clip(t / .08, 0, 1) * np.clip((d - t) / .08, 0, 1) * (.75 + .25 * np.sin(2 * np.pi * 11 * t))
    return x * e * g


def zap(g=1.0):
    t = T(.9)
    f = 180 + 3400 * np.exp(-t * 9)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR + 2.2 * np.sin(2 * np.pi * 73 * t))
    return x * np.exp(-t * 4.6) * g


def hum(d, g=1.0):
    t = T(d)
    x = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((110, .6), (220, .3), (331, .16), (880, .05)))
    e = np.clip(t / .05, 0, 1) * np.clip((d - t) / .3, 0, 1) * (.85 + .15 * np.sin(2 * np.pi * 31 * t))
    return x * e * g


def impact(g=1.0):
    t = T(3.4)
    f = 34 + 150 * np.exp(-t * 14)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.5)
    n = flt(rng.normal(0, 1, len(t)), "low", 2400) * np.exp(-t * 7) * .7
    c = flt(rng.normal(0, 1, len(t)), "high", 3000) * np.exp(-t * 34) * .5
    return np.tanh((b * 1.5 + n + c) * 1.3) * g


def buzz(d, f0=205, g=1.0):
    t = T(d)
    f = f0 + 6 * np.sin(2 * np.pi * 3.1 * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(ph * h + h) / h for h in (1, 2, 3, 4, 6))
    x = x + .4 * sum(np.sin(ph * 1.014 * h) / h for h in (1, 2, 3))
    n = flt(rng.normal(0, 1, len(t)), "band", (300, 2400)) * .22
    return flt(x * .36 + n, "low", 2600) * g


def flyby(d=1.8, g=1.0):
    """Rotor pass: pitch falls through the middle, panned left to right."""
    t = T(d)
    u = t / d
    f = 250 - 90 * np.tanh((u - .5) * 7)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(ph * h + h) / h for h in (1, 2, 3, 4, 6))
    n = flt(rng.normal(0, 1, len(t)), "band", (300, 3000)) * .5
    a = np.exp(-((u - .5) / .23) ** 2)
    m = (x * .4 + n) * a * g
    pan = np.clip((u - .5) * 2.2, -1, 1)
    return np.stack([m * np.cos((pan + 1) * np.pi / 4), m * np.sin((pan + 1) * np.pi / 4)], 1) * 1.41


def crackle(d, g=1.0):
    n = int(d * SR)
    x = np.zeros(n)
    for k in range(int(d * 90)):
        i = rng.integers(0, n - 800)
        a = rng.uniform(.2, 1) * (1 - i / n) ** 1.5
        L = rng.integers(60, 700)
        x[i:i + L] += rng.normal(0, 1, L) * np.hanning(L) * a
    return flt(x, "high", 1800) * g


def whine(d, g=1.0):
    t = T(d)
    f = 860 * np.exp(-t * 1.1) + 110
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + .3 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    return x * np.clip(t / .08, 0, 1) * np.clip((d - t) / .5, 0, 1) * g


# ---------- arrangement ----------
Dm = [73.4, 110, 146.8, 174.6]
Bb = [58.3, 116.5, 146.8, 174.6]
Gm = [49, 98, 146.8, 196]
Am = [55, 110, 164.8, 220]
PROG = [Dm, Dm, Bb, Bb, Gm, Gm, Am, Am]

# low drone under everything
t = T(DUR + .5)
dr = (np.sin(2 * np.pi * 36.7 * t) * .7 + np.sin(2 * np.pi * 55.05 * t + .4) * .3) * (1 + .15 * np.sin(2 * np.pi * .13 * t))
rum = flt(rng.normal(0, 1, len(t)), "low", 110) * .6
put(music, dr * .10 + rum * .05, 0)

# hook
put(music, sweep_noise(7.1, 90, 1800, .8, .98), .1, .10, 0)
put(fx, buzz(2.4, 190, .05) * np.clip(T(2.4) / 2.4, 0, 1), 0.0, 1, .4)
put(fx, flyby(1.8, .42), 2.4)
for tt, g in ((0.0, .7), (2.4, .8), (4.2, .9)):
    put(drums, taiko(g), tt)
    put(wet, taiko(g), tt, .35)
for k in range(20):       # the counter ticking up to ~20
    put(fx, tick(.5), 5.42 + 1.2 * (1 - (1 - k / 20) ** 1.25) * .95, 1)
put(fx, sweep_noise(1.2, 300, 6000, .7, .95), 6.0, .22)

# reveal
put(music, braam(73.4, 4.2, .75), 7.2)
put(wet, braam(73.4, 4.2, .6), 7.2)
put(drums, impact(.8), 7.2)

# parts: pad + pulse + transition hits
for k in range(4):
    tt = 9.6 + k * BAR
    put(music, pad(PROG[k * 2], BAR + .4, .32, 700 + k * 180), tt - .1)
    put(drums, taiko(.55), tt)
    put(fx, sweep_noise(.5, 500, 4500, .7, .8), tt - .32, .16, .4 if k % 2 else -.4)
    for j in range(16):
        if j % 4 == 0:
            put(drums, kick(.4), tt + j * BEAT / 4)
        put(music, saw(36.7 * 2 if k < 2 else (29.1 * 2 if k == 2 else 24.5 * 2), T(.13), 12, np.full(int(.13 * SR), 420 + k * 90)) * np.exp(-T(.13) * 14), tt + j * BEAT / 4, .16)
    for j in range(6):
        put(fx, beep(1500 + 400 * (j % 3), .03, .05), tt + .18 + j * .08, 1, -.3 + j * .12)

# scan: strip back to pings
put(music, pad(Dm, 2.6, .10, 380), 19.2)
for tt in (19.5, 20.7):
    p = ping(1320, .34)
    put(fx, p, tt)
    put(wet, p, tt, .7)
    put(fx, p * .3, tt + .3, 1, .5)
    put(fx, p * .14, tt + .6, 1, -.5)

# detect
put(music, pad(Bb, 4.9, .16, 500), 21.5)
put(fx, ping(1320, .3), 21.9)
put(fx, buzz(3.6, 190, .035) * np.clip(T(3.6), 0, 1) * np.clip(3.6 - T(3.6), 0, 1), 22.7, 1, .3)
for k in range(9):        # amber acquire beeps
    put(fx, beep(1760, .06, .16), 22.76 + k * .18)
put(drums, taiko(.8), 22.76)
put(wet, taiko(.8), 22.76, .3)
put(fx, beep(2349, .55, .16), 24.36)
put(drums, kick(.7), 24.36)
for tt in (24.96, 25.56, 26.16):
    put(drums, kick(.5), tt)

# pipeline: a tick per stage over a tightening pulse
for k, tt in enumerate((26.4, 28.0, 28.15, 29.3, 30.2)):
    put(fx, sweep_noise(.22, 900, 5000, .6, .4), tt - .05, .13, .3)
    put(fx, beep(880 * 2 ** (k * 2 / 12), .09, .16), tt)
for k in range(9):        # sensor frames clicking through while the eyes split
    put(fx, tick(.3), 26.5 + k * .15, 1, -.2 + k * .05)
for j in range(32):
    tt = 26.4 + j * BEAT / 2
    if j % 2 == 0:
        put(drums, kick(.5), tt)
    put(drums, hat(.05), tt + BEAT / 4)
put(music, pad(Gm, 4.9, .2, 800), 26.3)

# stereo
put(music, pad(Am, 4.9, .2, 900), 31.1)
put(drums, taiko(.6), 31.2)
put(drums, taiko(.6), 32.4)
put(fx, sweep_noise(1.0, 400, 3200, .7, .85), 32.8, .24)
put(drums, taiko(.8), 34.5)
put(wet, taiko(.8), 34.5, .3)
put(fx, beep(1174, .4, .13), 34.7)

# dash: full drive, four bars, filter opening
for bar in range(4):
    tt0 = 36.0 + bar * BAR
    put(music, pad(PROG[bar * 2], BAR + .3, .3, 900 + bar * 260), tt0 - .1)
    for j in range(16):
        tt = tt0 + j * BEAT / 4
        if j % 4 == 0:
            put(drums, kick(.85), tt)
        if j % 8 == 4:
            put(drums, clap(.42), tt)
            put(wet, clap(.42), tt, .22)
        put(drums, hat(.07 if j % 2 else .04), tt, 1, .15)
        if j in (3, 6, 11, 14):
            put(drums, kick(.45), tt)
        root = (36.7, 29.1, 24.5, 27.5)[bar] * 2
        put(music, saw(root, T(.14), 14, np.full(int(.14 * SR), 500 + bar * 160)) * np.exp(-T(.14) * 12), tt, .24)
    put(drums, taiko(.6), tt0)
for bar in range(4):
    put(fx, ping(1760, .1, .5), 36.2 + bar * BAR)

# command
put(music, pad(Dm, 4.9, .2, 1000), 45.5)
for j in range(16):
    tt = 45.6 + j * BEAT / 2
    put(drums, kick(.6 if j % 2 == 0 else .3), tt)
    put(drums, hat(.05), tt + BEAT / 4)
for j in range(34):
    put(fx, beep(1200 + 700 * ((j * 7) % 5) / 4, .022, .08), 45.7 + j * .052, 1, .3 * np.sin(j))
put(fx, servo(1.7, .16), 47.2)
put(fx, zap(.42), 48.9)
put(wet, zap(.42), 48.9, .5)
put(fx, hum(1.5, .09), 48.95)
put(drums, taiko(.85), 48.9)

# climax: pull everything out, riser, then the hit
put(music, sweep_noise(3.0, 150, 9000, .75, .98), 50.4, .34)
for k in range(5):
    tt = 50.4 + k * BEAT
    put(drums, kick(.55 + k * .08), tt)
    put(fx, beep(2349, .05, .1 + k * .02), tt + .02)
put(music, pad(Am, 3.1, .22, 1500), 50.3)
bt = T(3.0)
put(fx, np.sin(2 * np.pi * np.cumsum(120 + 90 * bt / 3.0) / SR) * (.6 + .4 * bt / 3.0) * .10, 50.4)
put(fx, hum(3.0, .10), 50.4)
put(fx, crackle(1.8, .34), 53.42)
put(wet, crackle(1.8, .3), 53.42, .35)
put(fx, whine(4.4, .10), 54.3, 1, -.3)
put(drums, impact(1.0), 53.4)
put(music, braam(73.4, 3.6, .85), 53.4)
put(wet, braam(73.4, 3.6, .8), 53.4)
put(wet, impact(.9), 53.4, .5)
put(fx, zap(.34), 53.38)

# montage
for k in range(4):
    tt = 55.2 + k * 1.2
    put(drums, taiko(1.0), tt)
    put(wet, taiko(1.0), tt, .35)
    put(fx, sweep_noise(.3, 1000, 6000, .7, .85), tt - .22, .15)
    put(music, pad(PROG[k * 2], 1.3, .34, 1500), tt - .05)
    for j in range(8):
        t8 = tt + j * .15
        put(drums, hat(.07), t8)
        if j % 4 == 0:
            put(drums, kick(.8), t8)
        if j == 4:
            put(drums, clap(.5), t8)

# kpi
put(music, pad(Dm, 2.5, .22, 800), 60.0)
put(music, pad(Bb, 2.5, .22, 900), 62.4)
put(music, pad(Am, 2.5, .22, 1000), 64.8)
put(drums, taiko(.7), 60.0)
for j in range(12):
    put(drums, kick(.5), 60.0 + j * BEAT)
for k in range(4):
    tt = 60.25 + k * .55
    put(fx, beep(880 * 2 ** ((0, 3, 7, 12)[k] / 12), .2, .15), tt)
    put(wet, beep(880 * 2 ** ((0, 3, 7, 12)[k] / 12), .2, .15), tt, .4)

# end
for tt, g in ((67.3, .8), (67.9, .9), (68.5, 1.1)):
    put(drums, taiko(g), tt)
    put(wet, taiko(g), tt, .4)
put(fx, sweep_noise(1.0, 200, 5000, .8, .95), 68.65, .24)
put(music, braam(73.4, 4.6, .8), 69.6)
put(wet, braam(73.4, 4.6, .9), 69.6)
put(drums, impact(.85), 69.6)
put(music, pad([73.4, 110, 146.8, 220], 4.4, .16, 700), 69.8)

# system voice
VO = {"scan": 19.45, "detect": 22.95, "lock": 24.45, "solve": 36.5, "cmd": 45.9, "laser": 49.3, "hit": 54.1}
for k, tt in VO.items():
    sr, x = wavfile.read(os.path.join(HERE, "vo", k + ".wav"))
    x = x.astype(np.float32) / 32768
    if x.ndim > 1:
        x = x.mean(1)
    x = resample_poly(x, SR, sr).astype(np.float32)
    x = x[np.abs(x).argmax() * 0 + np.argmax(np.abs(x) > .01):]
    x = flt(flt(x, "high", 320), "low", 3600)
    x = np.tanh(x / (np.abs(x).max() + 1e-9) * 2.2) * .8
    x = x * (1 + .12 * np.sin(2 * np.pi * 42 * np.arange(len(x)) / SR))
    put(vo, x, tt, {"hit": .86, "laser": .72, "solve": .66}.get(k, .5))
    put(wet, x, tt, .3)
    put(fx, beep(1650, .025, .1), tt - .07)

# ---------- mix ----------
ir_t = T(2.4)
ir = rng.normal(0, 1, (len(ir_t), 2)) * np.exp(-ir_t * 2.9)[:, None]
ir = flt(ir, "low", 5200)
rev = np.stack([fftconvolve(wet[:, c], ir[:, c])[:N] for c in (0, 1)], 1).astype(np.float32)
rev /= np.abs(rev).max() + 1e-9

# duck music under the voice
env = np.abs(vo).max(1)
env = np.maximum.accumulate(env[::-1])[::-1]
k = int(.12 * SR)
env = np.convolve(np.abs(vo).max(1), np.ones(k) / k, "same")
duck = (1 - np.clip(env * 6, 0, .62))[:, None]

mix = music * duck * .8 + drums * .78 * (duck * .6 + .4) + fx * (duck * .5 + .5) * .9 + vo * 1.0 + rev * (duck * .5 + .5) * .42
mix = flt(mix, "high", 26)
mix = np.tanh(mix * 1.15)
fade = np.ones(N)
e = int(DUR * SR)
fade[e - int(1.2 * SR):e] = np.linspace(1, 0, int(1.2 * SR)) ** 1.5
fade[e:] = 0
mix = (mix.T * fade).T[:e]
mix = mix / np.abs(mix).max() * .78
wavfile.write(os.path.join(HERE, "track.wav"), SR, (mix * 32767).astype(np.int16))
print("written", len(mix) / SR, "s; rms", float(np.sqrt((mix ** 2).mean())))
