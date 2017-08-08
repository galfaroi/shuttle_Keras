# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from keras.regularizers import l2
from keras.utils import np_utils
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#normalization of data for neural nets
def preprocess(raw_X):
    from sklearn import preprocessing

    # Scaled data has zero mean and unit variance:
    X = preprocessing.scale(raw_X)
    return X

#import dataset
names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'y']
data = pd.read_table('../shuttle.trn', sep='\s+', header=None, names=names)

#check columns and data
data.columns
data['y'].head(5)
data.head(5)
y=data['y']
y.head(5)

data.columns
data.head(5)
del data['y']
data.head(5)
#create ndarray
data.shape
y.shape
data = data.values
data

x=preprocess(data)
x.shape
y.shape
x[1]
y= np_utils.to_categorical(y)


# create model
model = Sequential()
model.add(Dense(36, input_dim=9, activation='relu'))
#model.add(Dropout(drop))
model.add(Dense(18, input_dim=9, activation='relu'))
model.add(Dense(8, activation='softmax'))
model.summary
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=20, batch_size=10)

#test data
test = pd.read_table('../shuttle.tst', sep='\s+', header=None, names=names)
test.head(5)
y_t = test['y']
y_t.head(5)
del test['y']

test.head(5)
x_test = preprocess(test)
x_test.shape

y_t.shape

y_t = y_t.values
y_t
y_t.shape
y_t = np_utils.to_categorical(y_t)
score = model.evaluate(x_test, y_t, batch_size=100)

score[0]
score[1]
print score
