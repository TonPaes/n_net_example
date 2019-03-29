import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import deep

new_model = tf.keras.models.load_model('num_reader.model')

predictions = new_model.predict(deep.x_test)

while 1:
    number = int(input())
    print(np.argmax(predictions[number]))

    plt.imshow(deep.x_test[number], cmap=plt.cm.binary)
    plt.show()
# comment to x_test
