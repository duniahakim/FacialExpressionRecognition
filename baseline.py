import pandas
import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout,Activation,Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import pickle

def baseline_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size = (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model

X_val = pickle.load( open( "X_val.p", "rb" ) )
Y_val = pickle.load( open( "Y_val.p", "rb" ) )
X_train = pickle.load( open( "X_train.p", "rb" ) )
Y_train = pickle.load( open( "Y_train.p", "rb" ) )

Y_train = np_utils.to_categorical(Y_train, num_classes=7)
Y_val = np_utils.to_categorical(Y_val, num_classes=7)

model = baseline_model()

result = model.fit(X_train, Y_train,
          batch_size=128,
          epochs=5,
          verbose=1,
          validation_data=(X_val, Y_val),
          shuffle=True)
