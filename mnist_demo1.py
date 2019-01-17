from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_lables), (test_images, test_lables) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # metrics监控指标

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)

network.fit(train_images, train_lables, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_lables)
print("test_acc:", test_acc)


# digit = train_images[4]
#
#
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()