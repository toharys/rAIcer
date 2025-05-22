// Include the Servo library
#include <Servo.h>

// Create a Servo object
Servo myServo;

// Variable to hold the servo angle
int angle = 0;

void setup() {
  // Attach the servo to pin D11
  myServo.attach(11);
}

void loop() {
  // Move from 0 degrees to 180 degrees in increments
  for (angle = 80; angle <= 130; angle += 5) {
    myServo.write(angle);    // Set the servo position
    delay(25);               // Short pause to allow the servo to reach position
  }

  // Move back from 180 degrees to 0 degrees
  for (angle = 130; angle >= 80; angle -= 5) {
    myServo.write(angle);
    delay(25);
  }
}
