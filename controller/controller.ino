#include <Wire.h>
#include <Servo.h>

// I2C slave address
#define I2C_SLAVE_ADDR 0x08

// Motor Driver Pins
const int STBY = 10;
const int AIN1 = 9;
const int AIN2 = 8;
const int PWMA = 5;

// Servo Pin
const int SERVO_PIN = 11;
Servo steeringServo;

// Steering Servo Range
// Servo range: 80° to 130°
const int SERVO_LEFT_ANGLE = 80;
const int SERVO_RIGHT_ANGLE = 130;
const int SERVO_CENTER = (SERVO_LEFT_ANGLE + SERVO_RIGHT_ANGLE) / 2; // 105

// Steering angle input range (example: -0.5 to +0.5 radians)
float STEERING_ANGLE_MIN = -0.5; 
float STEERING_ANGLE_MAX =  0.5;

// Speed range: -1.0 to +1.0
float SPEED_MIN = -1.0;
float SPEED_MAX =  1.0;

// Global variables to hold received commands
float receivedSpeed = 0.0;
float receivedSteering = 0.0;
volatile bool newCommandReceived = false;

void setup() {
  steeringServo.attach(SERVO_PIN);

  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);

  digitalWrite(STBY, HIGH);
  steeringServo.write(SERVO_CENTER);
  stopMotor();

  Wire.begin(I2C_SLAVE_ADDR);
  Wire.onReceive(receiveEvent);
}

void loop() {
  if (newCommandReceived) {
    noInterrupts();
    float speed = receivedSpeed;
    float steering = receivedSteering;
    newCommandReceived = false;
    interrupts();

    // Process steering
    int servoAngle = mapSteeringAngleToServo(steering);
    steeringServo.write(servoAngle);

    // Process speed
    setMotorSpeed(speed);
  }

  delay(10);
}

void receiveEvent(int howMany) {
  if (howMany >= 8) {
    union {
      float f;
      byte b[4];
    } speedData, steerData;

    for (int i = 0; i < 4; i++) {
      speedData.b[i] = Wire.read();
    }
    for (int i = 0; i < 4; i++) {
      steerData.b[i] = Wire.read();
    }

    noInterrupts();
    receivedSpeed = speedData.f;
    receivedSteering = steerData.f;
    newCommandReceived = true;
    interrupts();
  } else {
    // Clear any leftover bytes if not enough data
    while (Wire.available()) {
      Wire.read();
    }
  }
}

int mapSteeringAngleToServo(float angle) {
  angle = constrain(angle, STEERING_ANGLE_MIN, STEERING_ANGLE_MAX);
  float normalized = (angle - STEERING_ANGLE_MIN) / (STEERING_ANGLE_MAX - STEERING_ANGLE_MIN);
  float servoAngle = SERVO_LEFT_ANGLE + normalized * (SERVO_RIGHT_ANGLE - SERVO_LEFT_ANGLE);
  return (int)servoAngle;
}

void setMotorSpeed(float speed) {
  speed = constrain(speed, SPEED_MIN, SPEED_MAX);

  if (speed > 0) {
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, LOW);
    int pwmVal = (int)(speed * 255.0);
    analogWrite(PWMA, pwmVal);
  } else if (speed < 0) {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, HIGH);
    int pwmVal = (int)(-speed * 255.0);
    analogWrite(PWMA, pwmVal);
  } else {
    stopMotor();
  }
}

void stopMotor() {
  analogWrite(PWMA, 0);
}
