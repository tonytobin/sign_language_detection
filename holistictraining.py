
import numpy as np
import os
from matplotlib import image, pyplot as plt
import time
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense
from tensorflow.keras.callbacks import TensorBoard


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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)


actions=np.array(['hello','hungry'])#actions
log_dir=os.path.join('logs')
tb_callbacks=TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val) ,epochs=20,callbacks=[tb_callbacks])
model.summary()
res=model.predict(x_test)

#model.save('action.h5')