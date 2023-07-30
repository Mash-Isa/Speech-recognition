# %%
import numpy as np 
import pandas as pd

import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import librosa
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import save_model

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# %% [markdown]
# # Description

# %% [markdown]
# ## Strategy

# %% [markdown]
# ## Sources

# %% [markdown]
# - EDA: https://www.kaggle.com/code/davids1992/speech-representation-and-data-exploration
# - Model: https://www.kaggle.com/code/leangab/tensorflow-speech-recognition-challenge
# - Command detection: https://www.kaggle.com/code/araspirbadian/voice-command-detection/notebook
# - Speech Recognition: https://www.kaggle.com/code/fathykhader/speech-recognition
# 

# %% [markdown]
# # Data Loading

# %%
train_audio_path = 'data/tensorflow-speech-recognition-challenge/train/audio'
labels = os.listdir(train_audio_path)
labels

# %%
# find count of each label and plot bar graph
no_of_recordings = []

for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
# plot
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()

# %% [markdown]
# # Preprocessing

# %%
all_wave = []
all_label = []

for label in labels[:6]:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 8000)
        
        if(len(samples) == 8000) : 
            all_wave.append(samples)
            all_label.append(label)

# %%
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)

# %%
y = np_utils.to_categorical(y, num_classes=len(labels))

# %%
all_wave = np.array(all_wave).reshape(-1,8000)
all_wave.shape

# %%
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size=0.2, random_state=42, shuffle=True)
x_te, x_val, y_te, y_val = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5, random_state=42, shuffle=True)

# %% [markdown]
# # Model

# %% [markdown]
# ## Initialization

# %%
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 

# %% [markdown]
# ## Training

# %%
history = model.fit(x_tr, y_tr, epochs=100, callbacks=es, batch_size=32, validation_data=(x_val, y_val))

# %%
plt.plot(history.history['loss'], label='train') 
plt.plot(history.history['val_loss'], label='test') 
plt.legend()
plt.show()

# %%
# Save the model to a file
save_model(model, "model.h5")

# %% [markdown]
# # Evaluation

# %%
predictions = model.predict(x=x_te, verbose=0)
print(y_te.shape, predictions.shape)

# %%
print(classification_report(y_te.argmax(axis=1), predictions.argmax(axis=1)))

# %%
confusion_matrix = confusion_matrix(y_te.argmax(axis=1), predictions.argmax(axis=1))
sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.show()


