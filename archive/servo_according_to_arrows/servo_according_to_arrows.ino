#include <Servo.h>

// Create a Servo object
Servo myServo;

// Variable to hold the servo angle
int angle = 90;  // Start at center position
const int stepSize = 5;  // How much to move per command
const int minAngle = 60; // Minimum servo angle
const int maxAngle = 120; // Maximum servo angle

void setup() {
  // Attach the servo to pin D11
  myServo.attach(11);
  myServo.write(angle); // Set initial position
 
  // Start serial communication
  Serial.begin(9600);
}

void loop() {
  // Check if data is available from Jetson
  if (Serial.available()) {
    char command = Serial.read(); // Read the command

    // Adjust servo angle based on command
    if (command == 'w') { // Move right
      angle += stepSize;
      if (angle > maxAngle) angle = maxAngle;
    }
    else if (command == 's') { // Move left
      angle -= stepSize;
      if (angle < minAngle) angle = minAngle;
    }

    // Move servo to the new angle
    myServo.write(angle);
   
    // Print the angle for debugging
    Serial.print("Servo angle: ");
    Serial.println(angle);
  }
}
