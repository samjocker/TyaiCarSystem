#include <Servo.h>
#include <ADCTouch.h>
Servo myservo; 
int angle = 90;

void setup() {
  Serial.begin(9600);
  myservo.attach(9);
  pinMode(12, INPUT);
  pinMode(4, OUTPUT);
  

}
void loop() {
  while(true){
    int val = ADCTouch.read(A0);
    if(val > 900){
      myservo.detach();
      digitalWrite(4,HIGH);
      delay(5000);
      myservo.attach(9);

      
    }else{
      digitalWrite(4,LOW);
    }
    
    if (digitalRead(12)) {
      if (Serial.available()) {
        angle = Serial.parseInt();
        Serial.println("haha");
      }
    } else {
      angle = map(analogRead(A3),0, 1023, 0 ,180);
      Serial.println("lolo");
    }
    myservo.write(angle);
  }
}