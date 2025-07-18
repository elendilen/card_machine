#include "rotation_module.h"
#include "ejector_module.h"

void setup(){
  Serial.begin(9600);
  initRotation();
  initEjector();
}

void loop(){
  if(Serial.available()){
    int recievedChar = Serial.read();
    if(recievedChar == 'r'){
      Serial.println("r");
      rotateCircle();
    }else if(recievedChar == 's' || recievedChar == 'R'){
      while (Serial.available() && !isDigit(Serial.peek())) {
          Serial.read(); // 扔掉非数字
      }
      int angle = Serial.parseInt();
      Serial.println(angle);
      if(angle >= 0 && angle <= 180){
        rotateToAngle(angle);
        delay(2000);
        if(recievedChar == 's'){
          ejectCard();
          //Serial.println("Ejected!");
        }
      }  
    }else if(recievedChar == 'q'){
        rotateToAngle(0);
    }
    Serial.println("END");
  }   
}


/*
  理论上传数据的模型如下
  loop(){
    python -> aduino r
    aduino -> python END
  }
  loop() for 90 times
  s 0
  s 45
  s 90
  s 135 
*/