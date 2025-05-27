import serial
import glob
import os
from evdev import InputDevice, categorize, ecodes
import threading

# Path to keyboard input event
KEYBOARD_PATH = "/dev/input/event2"  # Adjust based on `cat /proc/bus/input/devices`

# Path to Arduino USB port
serial_ports = glob.glob('/dev/serial/by-id/*')

def initial_connection():
    global ser, keyboard

    if len(serial_ports) == 0:
        raise Exception("No serial device found. Check the connection.")

    # Resolve the symbolic link to get the actual port path
    ARDUINO_PORT = os.path.realpath(serial_ports[0])
    print(f"Using Arduino port: {ARDUINO_PORT}")
    BAUD_RATE = 9600

    # Connect to Arduino
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to Arduino on {ARDUINO_PORT}")
    except Exception as e:
        print(f"Arduino connection error: {e}")
        exit(1)

    # Connect to Keyboard
    try:
        keyboard = InputDevice(KEYBOARD_PATH)
        # Modern way to get device info
        print(f"Listening for key presses on device: {keyboard.name}")
        print(f"Device info: {keyboard.info}")
    except Exception as e:
        print(f"Keyboard connection error: {e}")
        exit(1)

def read_arduino():
    while True:
        try:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()
                if data:
                    print(f"Arduino: {data}")
        except Exception as e:
            print(f"Arduino read error: {e}")
            break

def communication():
    try:
        for event in keyboard.read_loop():
            if event.type == ecodes.EV_KEY and event.value == 1:
                key_event = categorize(event)
                keycode = key_event.keycode

                if keycode == "KEY_RIGHT":
                    print("Sending: w (Move Right)")
                    ser.write(b'w')
                elif keycode == "KEY_LEFT":
                    print("Sending: s (Move Left)")
                    ser.write(b's')
                elif keycode == "KEY_UP":
                    print("Sending: u (Move Forward)")
                    ser.write(b'u')
                elif keycode == "KEY_DOWN":
                    print("Sending: b (Move Backward)")
                    ser.write(b'b')
                elif keycode == "KEY_SPACE":
                    print("Sending: x (Stop)")
                    ser.write(b'x')
    except Exception as e:
        print(f"Keyboard communication error: {e}")
    finally:
        ser.close()
        print("Serial connection closed")

if __name__ == "__main__":
    initial_connection()
    threading.Thread(target=read_arduino, daemon=True).start()
    communication()
