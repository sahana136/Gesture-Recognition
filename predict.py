import numpy as np
import tensorflow as tf
from pynput.keyboard import Key, Controller
import operator
import cv2
import sys, os
import time
import pyautogui

keyboard = Controller()

# Loading the model
jsonFile = open("model.json", "r")
modelJson = jsonFile.read()
jsonFile.close()
loadedModel =tf.keras.models.model_from_json(modelJson)
# load weights into new model
loadedModel.load_weights("model.h5")

cap = cv2.VideoCapture(0)

# Category dictionary
Category = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'}

while True:
    _, frame = cap.read()
    # mirror image
    frame = cv2.flip(frame, 1)
    # Coordinates of the ROI
    m = int(0.5*frame.shape[1])
    n = 10
    m1 = frame.shape[1]-10
    n1 = int(0.5*frame.shape[1])
    # Drawing the ROI
  
    cv2.rectangle(frame, (m-1, n-1), (m1+1, n1+1), (255,0,0) ,1)
    # Extracting the ROI
    ROI = frame[n:n1, m:m1]
    
    # Resizing the ROI for prediction
    ROI = cv2.resize(ROI, (64, 64)) 
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    _, testImage = cv2.threshold(ROI, 123, 230, cv2.THRESH_BINARY)
    cv2.imshow("test", testImage)
    res = loadedModel.predict(testImage.reshape(1, 64, 64, 1))
    prediction = {'Zero': res[0][0], 
                  'One': res[0][1], 
                  'Two': res[0][2],
                  'Three': res[0][3],
                  'Four': res[0][4],
                  'Five': res[0][5]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    cv2.putText(frame, prediction[0][0], (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1) 
    cv2.putText(frame, "Zero-SWITCHWINDOW", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "One-SCROLLUP", (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Two-SCROLLDOWN", (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Four-SCREENSHOT", (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "Five-PLAY/PAUSE", (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # Displaying the predictions
    if prediction[0][0] == 'Five':
    	keyboard.press(Key.space)
    if prediction[0][0] =='ZERO':
    	pyautogui.keyDown('alt')
    	pyautogui.keyDown('tab')
    	
    	pyautogui.keyUp('tab')
    	pyautogui.keyUp('alt')
    if prediction[0][0] =='One':
    	pyautogui.press('up')
    if prediction[0][0]=='Two':
    	pyautogui.press('down')

    if prediction[0][0]=='Three':
    	pass
    if prediction[0][0]=='Four':
    	pyautogui.press('prtscr')

    cv2.imshow("Frame", frame)
    interrupt = cv2.waitKey(5)
    if interrupt & 0xFF == 27: # esc key
    	break
        
 
cap.release()
cv2.destroyAllWindows()
