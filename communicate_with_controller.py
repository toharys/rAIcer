import serial
from evdev import InputDevice, categorize, ecodes

# Path to keyboard input event
KEYBOARD_PATH = "/dev/input/event2"  # Adjust based on `cat /proc/bus/input/devices`

# Path to Arduino USB port
ARDUINO_PORT = "/dev/ttyUSB0"  # Check using `ls /dev/serial/by-id/`
BAUD_RATE = 9600

# Connect to Arduino
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Connect to Keyboard
try:
    keyboard = InputDevice(KEYBOARD_PATH)
    print(f"Listening for key presses on {keyboard.path}...")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Function to continuously read Arduino responses
def read_arduino():
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            if data:
                print(f"Arduino: {data}")

# Start a background thread to read Arduino messages
import threading
threading.Thread(target=read_arduino, daemon=True).start()


# Read keyboard input and send to Arduino
for event in keyboard.read_loop():
    if event.type == ecodes.EV_KEY and event.value == 1:  # Key press event
        key_event = categorize(event)

        if key_event.keycode == "KEY_RIGHT":
            print("Sending: w (Move Right)")
            ser.write(b'w')  # Send 'w' to Arduino

        elif key_event.keycode == "KEY_LEFT":
            print("Sending: s (Move Left)")
            ser.write(b's')  # Send 's' to Arduino

        elif key_event.keycode == "KEY_UP":
            print("Sending: u (Move Forward)")
            ser.write(b'u')  # Send 'u' to Arduino

        elif key_event.keycode == "KEY_DOWN":
            print("Sending: b (Move Backward)")
            ser.write(b'b')  # Send 'b' to Arduino

        elif key_event.keycode == "KEY_SPACE":
            print("Sending: x (Stop)")
            ser.write(b'x')  # Send 'x' to Arduino
