#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>
Adafruit_PWMServoDriver pwm;

#define NUM_SERVOS 1
#define SERVO_MIN 125
#define SERVO_MAX 550

int servoPins[NUM_SERVOS] = {0};
int currentServoPosition = SERVO_MIN;

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(50);  // Set the PWM frequency for the PCA9685

  Serial.println("Init max.");

  // Range check all servos to their maximum positions
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(servoPins[i], 0, SERVO_MAX);
  }

  delay(3000);

  Serial.println("Init min.");

  // Initialize all servos to their minimum positions
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(servoPins[i], 0, SERVO_MIN);
  }

  delay(3000);

  Serial.println("Send a value between 0 and 1 to set the servo position.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n'); // Read the input until newline
    float position = input.toFloat(); // Convert input to a float
    
    // Ensure the position is within the valid range
    if (position >= 0.0 && position <= 1.0) {
      currentServoPosition = map(position * 1000, 0, 1000, SERVO_MIN, SERVO_MAX); 
      pwm.setPWM(servoPins[0], 0, currentServoPosition);
      Serial.print("Servo Position: ");
      Serial.println(currentServoPosition); // Send position data for plotting
    } else {
      Serial.println("Invalid input. Please send a value between 0 and 1.");
    }
  }

  // Plot the current position for the Serial Plotter
  Serial.println(currentServoPosition);
  delay(100);
}
