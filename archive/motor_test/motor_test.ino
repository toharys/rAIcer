const int STBY = 10; // Make sure pin 4 on Hamster is really a GPIO pin you can set high
const int AIN1 = 9;
const int AIN2 = 8;
const int PWMA = 5; // Ensure this pin supports PWM on the Hamster board

void setup() {
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);

  digitalWrite(STBY, HIGH); // Enable driver
}

void loop() {
  // Forward full speed
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 255);
  delay(3000);

  // Stop
  analogWrite(PWMA, 0);
  delay(1000);

  // Reverse half speed
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, HIGH);
  analogWrite(PWMA, 128);
  delay(3000);

  // Stop
  analogWrite(PWMA, 0);
  delay(1000);
}
