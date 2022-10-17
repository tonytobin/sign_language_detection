
from ast import Num
import cv2
import numpy as np
import os
from matplotlib import image, pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense
from tensorflow.keras.callbacks import TensorBoard
import subprocess





#for mp holistic
mp_holostic = mp.solutions.holistic
#mp drawing
mp_drawing=mp.solutions.drawing_utils
DATA_PATH=os.path.join('MP_Data') #data path
actions=np.array(['hello','hungry'])#actions
no_sequences=30#number of videos
sequence_lenght=30#frames
sequence=[]
sentence=[]
predictions=[]
output=[]
threshold=0.4

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

model.load_weights('action.h5')

def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_landmark(image,results):
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holostic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holostic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holostic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(80,110,10),thickness=0,circle_radius=1))
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holostic.POSE_CONNECTIONS)

# def draw_landmarkadv(image,results):
#     mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holostic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holostic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holostic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(80,110,10),thickness=0,circle_radius=1))
#     mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holostic.POSE_CONNECTIONS)    

def extract_keypoints(results):
    lefthand=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    righthand=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    return np.concatenate([lefthand,righthand,face,pose])

cap =cv2.VideoCapture(0)
with mp_holostic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        img_dbg = np.zeros([640,480,1],dtype=np.uint8)
        img_dbg.fill(255)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_landmark(image, results)
        # draw_landmarkadv(image,results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            print(sentence[1])
                            subprocess.Popen('python /Applications/friday/speak.py '+sentence[1],shell=True)

                    else:
                        sentence.append(actions[np.argmax(res)])
                        

            if len(sentence) > 1: 
                sentence = sentence[-1:]
        output=str(sentence).strip("[]''")
        # cv2.rectangle(image, (300,600), (600, 440), (245, 117, 16), -1)
        cv2.rectangle(image, (5, 600), (400, 700), (240, 240, 240), cv2.FILLED)
        cv2.putText(image,str(sentence).strip("[]''"), (20,650), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        # cv2.imshow('advanced details', img_dbg)
        if cv2.waitKey(1) & 0xff ==ord('q') :
            break
    cap.release()
    cv2.destroyAllWindows()    