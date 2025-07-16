#include <Servo.h>

// Servo Configuration
const int SERVO_PIN = 11;
Servo myServo;
const int stepSize = 5;
const int minAngle = 40;
const int maxAngle = 120;
int angle = (minAngle + maxAngle) / 2;  // Center position

// Motor Driver Pins
const int STBY = 10;
const int AIN1 = 9;
const int AIN2 = 8;
const int PWMA = 5;

// Motor Timing Control
unsigned long motorStopTime = 0;
bool motorActive = false;
const int PULSE_DURATION = 30;  // ms for short forward bursts
const float PULSE_POWER = 1.0;  // 80% power for short bursts

void setup() {
  // Servo initialization
  myServo.attach(SERVO_PIN);
  myServo.write(angle);

  // Motor control setup
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  digitalWrite(STBY, HIGH);

  Serial.begin(9600);
}

void loop() {
  // Handle motor pulse timeout
  
  if (motorActive && millis() >= motorStopTime) {
    stopMotor();
    motorActive = false;
  }

  // Process serial commands
  if (Serial.available()) {
    char command = Serial.read();
    while (Serial.available()) Serial.read();  // Clear buffer

    Serial.print("Received: ");
    Serial.println(command);

    // Steering control
    if (command == 'w' || command == 's') {
      // Update steering angle
      angle += (command == 'w') ? stepSize : -stepSize;
      angle = constrain(angle, minAngle, maxAngle);
      myServo.write(angle);
      Serial.println("Angle:" + String(angle));
      // Start timed forward pulse if not already active
      //if (!motorActive) {
      //  moveMotor(PULSE_POWER);
      //  motorStopTime = millis() + PULSE_DURATION;
      //  motorActive = true;
      //}
    }
    // Direct motor control
    else if (command == 'u') {  // Forward
      moveMotor(1.0);
      motorActive = false;  // Cancel any timed pulse
    }
    else if (command == 'b') {  // Backward
      moveMotor(-1.0);
      motorActive = false;
    }
    else if (command == 'x') {  // Stop
      stopMotor();
      motorActive = false;
    }
  }
}

void moveMotor(float speed) {
  speed = constrain(speed, -1.0, 1.0);
  int pwmVal = abs(speed) * 255;

  digitalWrite(AIN1, speed > 0 ? HIGH : LOW);
  digitalWrite(AIN2, speed > 0 ? LOW : HIGH);
  analogWrite(PWMA, pwmVal);
}

void stopMotor() {
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 0);
}