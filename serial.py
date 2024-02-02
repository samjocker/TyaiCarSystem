import serial
import time
import os

openSerial = False

def animate_rocket():
  distance_from_top = 20
  for i in range(20):
    print("\n" * distance_from_top)
    print("          /\        ")
    print("          ||        ")
    print("          ||        ")
    print("         /||\        ")
    time.sleep(0.2)
    os.system('clear')  
    distance_from_top -= 1
    if distance_from_top < 0:
      distance_from_top = 20

if openSerial:
    print("Wait connect")
    COM_PORT = '/dev/cu.usbmodem1101'
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    print("Connect successfuly")
    symbols = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
    for i in range(20):
        text = "<"
        for j in range(i):
            text += symbols[i%8]
        for j in range(20-i):
            text += " "
        text += ">"
        print(text, end='\r')
        time.sleep(0.1)
    print("Auto Pilot start!!!")
    ser.write((str(90)).encode())
    print("servo 90 degress")
    # time.sleep(5)
    # print("servoFree!!!")
