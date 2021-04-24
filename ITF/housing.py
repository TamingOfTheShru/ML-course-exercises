# A simple model predict output based on the defined relatioship as follows.
# Cost of house = 50$ + number of bedrooms. For example, one bedroom apartment would cost $100, 2 bedroom apartment would cost $150 and so on.



import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype=float)
ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0], dtype=float)
model.fit(xs,ys,epochs=700)
print(model.predict([7.0]))
