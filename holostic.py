import cv2
import numpy as np
import os
from matplotlib import image, pyplot as plt
import time
import mediapipe as mp

#for mp holistic
mp_holostic = mp.solutions.holistic
#mp drawing
mp_drawing=mp.solutions.drawing_utils
DATA_PATH=os.path.join('MP_Data') #data path
actions=np.array(['hello','hungry'])#actions
no_sequences=30#number of videos
sequence_lenght=30#frames

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
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holostic.FACEMESH_TESSELATION,mp_drawing.DrawingSpec(color=(80,110,10),thickness=0,circle_radius=0))
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holostic.POSE_CONNECTIONS)

# def extract_keypoints(results):
#     lefthand=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     righthand=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
#     pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
#     return np.concatenate([lefthand,righthand,face,pose])


cap =cv2.VideoCapture(0)
with mp_holostic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        #get image
        ret,frame =cap.read()
        image,results = mediapipe_detection(frame,holistic)

        draw_landmark(image,results)
        

        cv2.imshow("img",image)
        if cv2.waitKey(1) & 0xff ==ord('q') :
            break
    cap.release()
    cv2.destroyAllWindows()    