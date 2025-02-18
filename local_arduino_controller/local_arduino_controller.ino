#include <Servo.h>

Servo myServo;  // Servo object
//int motorPin = 9;  // Motor control PWM pin
int servoAngle = 90;  // Default servo position (neutral)

// Motor control pins
const int STBY = 10;  // Standby pin
const int AIN1 = 9; // Motor direction
const int AIN2 = 8;  
const int PWMA = 5; // PWM control

void setup() {
    Serial.begin(9600);  // Start serial communication for reading keyboard input

    // Servo setup
    myServo.attach(3);  // Connect servo to pin 3
    myServo.write(servoAngle);

    // Motor setup
    pinMode(STBY, OUTPUT);
    pinMode(AIN1, OUTPUT);
    pinMode(AIN2, OUTPUT);
    pinMode(PWMA, OUTPUT);

    digitalWrite(STBY, HIGH); // Enable motor driver
    
    Serial.println("Arduino Ready: Use W/S (Servo), U/B (Motor), X (Stop)");
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read();

        // Ignore carriage return and newline
        if (command == '\n' || command == '\r') {
            return;
        }

        Serial.print("Received Command: ");
        Serial.println(command);

        switch (command) {
            case 'w':  // Turn servo right
                servoAngle = min(servoAngle + 10, 180);
                myServo.write(servoAngle);
                Serial.print("Servo Right: ");
                Serial.println(servoAngle);
                break;

            case 's':  // Turn servo left
                servoAngle = max(servoAngle - 10, 0);
                myServo.write(servoAngle);
                Serial.print("Servo Left: ");
                Serial.println(servoAngle);
                break;

            case 'u':  // Move motor forward
                moveMotor(true, 255);
                Serial.println("Motor Forward");
                break;

            case 'b':  // Move motor backward
                moveMotor(false, 128);
                Serial.println("Motor Backward");
                break;

            case 'x':  // Stop motor
                stopMotor();
                Serial.println("Motor Stopped");
                break;

            default:
                Serial.print("Invalid Command: ");
                Serial.print(command);
                Serial.println(". Use W/S/U/B/X.");

                break;
        }
    }
}

void moveMotor(bool forward, int speed) {
    if (forward) {
        digitalWrite(AIN1, HIGH);
        digitalWrite(AIN2, LOW);
    } else {
        digitalWrite(AIN1, LOW);
        digitalWrite(AIN2, HIGH);
    }
    analogWrite(PWMA, speed);
}

void stopMotor() {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    analogWrite(PWMA, 0);
}
