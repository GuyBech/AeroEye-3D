# AeroEye-3D promo render. Each chunk runs in its own process; `final` joins them and adds the sound.
#   python render.py chunk <first_frame> <last_frame> <out.mkv>
#   python render.py all [workers]
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FF = glob.glob(os.path.join(HERE, "..", "tools", "iff", "imageio_ffmpeg", "binaries", "*.exe"))[0]
FPS = 30
TOTAL = int(74.4 * FPS)


def chunk(a, b, out):
    import cv2
    cv2.setNumThreads(2)
    import scenes2 as scenes
    p = subprocess.Popen([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", "1920x1080", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", "11", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(a, b):
        p.stdin.write(scenes.frame(f).tobytes())
        if (f - a) % 60 == 0:
            print(f"[{a}-{b}] {f} {(time.time() - t0) / max(1, f - a):.2f} s/f", flush=True)
    p.stdin.close()
    p.wait()


def final(n):
    lst = os.path.join(HERE, "chunks.txt")
    with open(lst, "w") as f:
        for k in range(n):
            f.write(f"file 'chunk_{k}.mkv'\n")
    out = os.path.join(HERE, "AeroEye-3D_Promo.mp4")
    subprocess.check_call([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                           "-i", os.path.join(HERE, "track.wav"), "-map", "0:v", "-map", "1:a",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                           "-profile:v", "high", "-level", "4.2", "-c:a", "aac", "-b:a", "256k",
                           "-movflags", "+faststart", "-shortest", out])
    print("final", out, os.path.getsize(out) / 1e6, "MB")


if __name__ == "__main__":
    if sys.argv[1] == "chunk":
        chunk(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "final":
        final(int(sys.argv[2]))
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        cuts = [TOTAL * k // n for k in range(n + 1)]
        procs = [subprocess.Popen([sys.executable, __file__, "chunk", str(cuts[k]), str(cuts[k + 1]),
                                   os.path.join(HERE, f"chunk_{k}.mkv")]) for k in range(n)]
        bad = [p.wait() for p in procs]
        print("chunks done", bad, flush=True)
        if not any(bad):
            final(n)
