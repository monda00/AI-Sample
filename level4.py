from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras.utils import np_utils

iris = load_iris()

X = iris['data']
y = np_utils.to_categorical(iris['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10)

score = model.evaluate(X_test, y_test)

print('Score is', score[1])
