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


def cleanData(dataset):
    Y = np.array(list(map(int, dataset[:, 0])))
    length = Y.shape[0]
    Y = Y.reshape(length, 1)
    X = np.zeros((length, 48, 48))
    for row in range(Y.shape[0]):
        X[row, :, :] = np.array(list(map(int, dataset[row, 1].split()))).reshape(48, 48)
    data = np.concatenate((Y, X.reshape(length, -1)), axis = 1)
    np.random.shuffle(data)
    Y = data[:, 0].reshape(length, 1)
    X = data[:, 1:].reshape(length, 48, 48)
    return X, Y

dataframe = pandas.read_csv("train.csv", header = 0)
dataset = dataframe.values
X, Y = cleanData(dataset)

X_val = X[:5741, :, :].reshape(-1, 48, 48, 1)
Y_val = Y[:5741, :]

X_test = X[5741:7741, :, :].reshape(-1, 48, 48, 1)
y_label = Y[5741:7741, :]

X_train = X[7741:, :, :].reshape(-1, 48, 48, 1)
Y_train = Y[7741:, :]

pickle.dump(X_train, open('X_train.p', 'wb'))
pickle.dump(Y_train, open('Y_train.p', 'wb'))
pickle.dump(X_val, open('X_val.p', 'wb'))
pickle.dump(Y_val, open('Y_val.p', 'wb'))
pickle.dump(X_test, open('X_test.p', 'wb'))
pickle.dump(y_label, open('y_label.p', 'wb'))
