import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import deep
import random


deep.train_net()

new_model = tf.keras.models.load_model('num_reader.model')

predictions = new_model.predict(deep.x_test)

while 1:
    input()
    number = random.randint(0, 2000)
    number = np.argmax(predictions[number])
    print(number)
    title = "Prediction: " + str(number)

    plt.title(title)
    plt.imshow(deep.x_test[number], cmap=plt.cm.binary)
    plt.show()
