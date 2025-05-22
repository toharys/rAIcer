from rAIcer.robot_control import Robot, Action , FrameStack
import time
import threading
import torch

class DataRecorder:
    def __init__(self, k=4):
        self.robot = Robot()
        self.frame_stack = FrameStack(k)
        self.recorded_data = []

        self.last_action = Action.STOP
        self.running = True
        self.stop_counter = 0 

        self.robot.connect_to_keyboard()

        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.listen_to_keyboard, daemon=True)
        self.keyboard_thread.start()

    def listen_to_keyboard(self):
        """Update the last action based on key presses."""
        key_action_map = {
            'KEY_RIGHT': Action.RIGHT,
            'KEY_LEFT': Action.LEFT,
            'KEY_UP': Action.FORWARD,
            'KEY_DOWN': Action.BACKWARD,
            'KEY_SPACE': Action.STOP,
        }

        for event in self.robot.keyboard.read_loop():
            if event.type == ecodes.EV_KEY and event.value == 1:
                keycode = categorize(event).keycode
                if keycode == 'KEY_Y':
                    self.running = False
                    return
                if keycode in key_action_map:
                    self.last_action = key_action_map[keycode]
                    self.robot.control_action(self.last_action)

    def record(self):
        """Main loop to record frames and actions."""
        print("Recording... Press Ctrl+C to stop.")
        try:
            first_frame = self.robot.get_grayscale_frame()
            if first_frame is None:
                print("Could not get initial frame")
                return

            # Initialize frame stack
            stacked = self.frame_stack.reset(first_frame)

            while self.running:
                new_frame = self.robot.get_grayscale_frame()
                if new_frame is None:
                    continue

                stacked = self.frame_stack.step(new_frame)
                # Save tuple: (frame_stack, action)
                self.recorded_data.append((stacked.clone(), self.last_action))

                time.sleep(0.05)  # Limit frame rate to ~20 FPS
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        finally:
            self.robot.close()

    def save(self, path="recorded_data.pt"):
        """Save the recorded data to a file"""
        print(f"Saving {len(self.recorded_data)} samples to {path}")
        torch.save(self.recorded_data, path)
        print("Done.")


if __name__ == "__main__":
    recorder = DataRecorder(k=4)
    recorder.record()
    recorder.save()
