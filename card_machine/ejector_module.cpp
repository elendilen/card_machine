#include "ejector_module.h"
#include <AFMotor.h>
#include <Arduino.h>

AF_DCMotor motor1(1);
int sensorPin = 15; //define analog pin 2


void initEjector() {
  motor1.setSpeed(150);
  motor1.run(RELEASE);
}

void ejectCard() {
  //Serial.write("ejecting card!\n");
  while(analogRead(sensorPin) < 400){
    motor1.run(FORWARD);
    delay(80);
  }
  motor1.run(RELEASE);
}
