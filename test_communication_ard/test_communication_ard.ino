int pressCount = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for serial to be ready (on some boards)
  Serial.println("Arduino Ready");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == ' ') {
      pressCount++;
    } else if (c == 'q') {
      Serial.print("Arduino count: ");
      Serial.println(pressCount);
    }
  }
}
