
import cv2
from cv2 import KeyPoint
import numpy as np
import os
from matplotlib import image, pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


DATA_PATH=os.path.join('MP_Data') #data path
actions=np.array(['hello','hungry'])#actions
no_sequences=30#number of videos
sequence_length=30#frames
label_map={label:num for num,label in enumerate(actions)}

sequences,labels =[],[]
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

x=np.array(sequences)
y=to_categorical(labels).astype(int)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05)
print(y_train)