#include <Servo.h>

// Create a Servo object
const int SERVO_PIN = 11;
Servo myServo;

// Motor Driver Pins
const int STBY = 10;
const int AIN1 = 9;
const int AIN2 = 8;
const int PWMA = 5;

// Speed range: -1.0 to +1.0
float SPEED_MIN = -1.0;
float SPEED_MAX =  1.0;

// Servo variables
const int stepSize = 5;
const int minAngle = 80;
const int maxAngle = 130;
int angle = (minAngle + maxAngle) / 2;  // Start at center position

void setup() {
  // Attach the servo to pin D11
  myServo.attach(SERVO_PIN);
  myServo.write(angle); // Set initial position

  // Motor setup
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  
  digitalWrite(STBY, HIGH); // Enable motor driver

  // Start serial communication
  Serial.begin(9600);
}

void loop() {
  // Check if data is available from Jetson
  if (Serial.available()) {
    char command = Serial.read(); // Read the command
    
    // Send the received command back to Jetson for debugging
    Serial.print("Received from Jetson: ");
    Serial.println(command);
    
    // Process Steering: Adjust servo angle
    if (command == 'w') { // Move right
      angle += stepSize;
      if (angle > maxAngle) angle = maxAngle;
    }
    else if (command == 's') { // Move left
      angle -= stepSize;
      if (angle < minAngle) angle = minAngle;
    }
    myServo.write(angle);
    
    // Send servo position update
    Serial.print("Servo angle: ");
    Serial.println(angle);

    // Motor control
    if (command == 'u') { // Move forward
      Serial.println("Motor moving forward");
      moveMotor(1.0);
    } 
    else if (command == 'b') { // Move backward
      Serial.println("Motor moving backward");
      moveMotor(-1.0);
    } 
    else if (command == 'x') { // Stop motor
      Serial.println("Motor stopped");
      stopMotor();
    }
    
    delay(50);
  }
}

void moveMotor(float speed) {
  speed = constrain(speed, SPEED_MIN, SPEED_MAX);
  int pwmVal = abs(speed) * 255;

  if (speed > 0) {
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, LOW);
  } else if (speed < 0){
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, HIGH);
  }
  analogWrite(PWMA, pwmVal);
}

void stopMotor() {
  analogWrite(PWMA, 0);
}
