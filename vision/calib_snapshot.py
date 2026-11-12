r"""
AeroEye-3D — save / restore the calibration as one snapshot.

Recalibrating replaces several files that only make sense together:
    calib3d.json            board model (the base map)
    aim_correction.json     residual patch, fitted RELATIVE to that base
    autocal_points.json     raw board-sweep points the base was fitted on
    ball_calib_points.json  ball points the patch was fitted on
A patch applied on top of a different base is silently wrong, so they are
saved and restored as a set. A snapshot is a folder in calib_backups/ named
<date>_<time>_<label>; restoring copies every file in it back, and removes an
active file the snapshot did not have (e.g. no correction at that time).

    ..\.venv\Scripts\python.exe calib_snapshot.py save before_recal
    ..\.venv\Scripts\python.exe calib_snapshot.py list
    ..\.venv\Scripts\python.exe calib_snapshot.py restore before_recal
"""
import os
import shutil
import sys
import time

FILES = ("calib3d.json", "aim_correction.json",
         "autocal_points.json", "ball_calib_points.json")
ROOT = "calib_backups"


def save(label):
    """Copy the active calibration into a new snapshot folder; return its path."""
    name = time.strftime("%Y%m%d_%H%M%S") + "_" + label
    dst = os.path.join(ROOT, name)
    os.makedirs(dst)
    saved = []
    for f in FILES:
        if os.path.exists(f):
            shutil.copy2(f, dst)
            saved.append(f)
    print(f"snapshot {name}: {', '.join(saved) or 'nothing to save'}", flush=True)
    return dst


def snapshots():
    if not os.path.isdir(ROOT):
        return []
    return sorted(d for d in os.listdir(ROOT)
                  if os.path.isdir(os.path.join(ROOT, d))
                  and any(os.path.exists(os.path.join(ROOT, d, f)) for f in FILES))


def find(label):
    hits = [d for d in snapshots() if d == label or d.endswith("_" + label)]
    return hits[-1] if hits else None          # newest match


def restore(label):
    name = find(label)
    if name is None:
        print(f"no snapshot matching '{label}'. Available:")
        list_()
        return False
    src = os.path.join(ROOT, name)
    save("auto_before_restore")                # restoring is itself undoable
    for f in FILES:
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, f)
            print(f"  restored {f}")
        elif os.path.exists(f):
            os.remove(f)
            print(f"  removed {f} (not part of that snapshot)")
    print(f"active calibration = {name}", flush=True)
    return True


def list_():
    for d in snapshots():
        have = [f for f in FILES if os.path.exists(os.path.join(ROOT, d, f))]
        print(f"  {d:45s} {', '.join(have)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("save", "restore", "list"):
        print(__doc__)
    elif a[0] == "list":
        list_()
    elif len(a) < 2:
        print("give a label, e.g.: calib_snapshot.py save before_recal")
    elif a[0] == "save":
        save(a[1])
    else:
        restore(a[1])
