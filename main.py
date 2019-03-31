import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import deep
import random


new_model = tf.keras.models.load_model('num_reader.model')

predictions = new_model.predict(deep.x_test)

while 1:
    input()
    number = random.randint(0, 1000)
    print(np.argmax(predictions[number]))

    plt.imshow(deep.x_test[number], cmap=plt.cm.binary)
    plt.show()
