#include <Servo.h>
#include <Arduino.h>
#include "rotation_module.h"

Servo servo1,servo2;
const int servoPin = 9;
const int servoPin2 = 10;
int current = 0;//当前的角度
void initRotation(){
  servo1.attach(servoPin);
  servo2.attach(servoPin2);
  servo1.write(0);
  servo2.write(0);
}

void rotateToAngle(int angle){
  if(current > angle){
    for(;current > angle;current-=2){
      servo1.write(current);
      servo2.write(current);
      delay(100);
    }
  }else{
    for(;current <= angle;current = current+2){
      servo1.write(current);
      servo2.write(current);
      delay(100);
    }
  }
}

void rotateCircle(){
  current+=2;
  servo1.write(current);
  servo2.write(current);
  //Serial.println(current);
}