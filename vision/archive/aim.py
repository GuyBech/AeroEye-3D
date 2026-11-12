r"""
AeroEye-3D — Phase 6a: PC-side UART control.

Sends "PAN,TILT,LASER\n" lines to the ESP32 and prints its acknowledgements.
Provides send() for other code (the vision loop will call this in Phase 6),
plus two ways to run standalone:

  demo:         ..\.venv\Scripts\python.exe aim.py [COM3]
      Sweeps pan then tilt through a few positions (laser OFF), proving the
      command channel end-to-end.
  interactive:  ..\.venv\Scripts\python.exe aim.py [COM3] --i
      Type "PAN,TILT,LASER" lines yourself (e.g. 120,80,0). Blank line = quit.

Safe range: 30..150 deg (the firmware clamps anyway). Laser defaults OFF.
"""
import sys
import time
import serial

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM3")
BAUD = 115200


def open_link(port=PORT):
    ser = serial.Serial(port, BAUD, timeout=1)
    time.sleep(2.0)              # ESP32 resets on port open; wait for boot
    ser.reset_input_buffer()
    return ser


def send(ser, pan, tilt, laser):
    """Send one command and return the ESP32's reply line."""
    ser.write(f"{int(pan)},{int(tilt)},{int(laser)}\n".encode())
    reply = ser.readline().decode(errors="replace").strip()
    print(f"  -> {pan},{tilt},{laser}   <- {reply}")
    return reply


def demo(ser):
    print("Demo: sweeping pan, then tilt (laser OFF).")
    for pan in (90, 120, 60, 90):
        send(ser, pan, 90, 0)
        time.sleep(0.8)
    for tilt in (90, 120, 60, 90):
        send(ser, 90, tilt, 0)
        time.sleep(0.8)
    print("Demo done.")


def interactive(ser):
    print("Interactive. Type PAN,TILT,LASER (e.g. 120,80,0). Blank line quits.")
    while True:
        try:
            line = input("cmd> ").strip()
        except EOFError:
            break
        if not line:
            break
        parts = line.split(",")
        if len(parts) != 3:
            print("  need 3 values: PAN,TILT,LASER")
            continue
        send(ser, *parts)


def main():
    ser = open_link()
    # drain the boot banner
    time.sleep(0.2)
    while ser.in_waiting:
        print("  [esp]", ser.readline().decode(errors="replace").strip())
    if "--i" in sys.argv:
        interactive(ser)
    else:
        demo(ser)
    ser.close()


if __name__ == "__main__":
    main()
