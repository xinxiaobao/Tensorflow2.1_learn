import tensorflow as tf 
from matplotlib import pyplot as plt 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0], cmap='gray')
plt.show()

print('x_train[0]:\n', x_train[0])
print('y_train[0]:\n', y_train[0])

print('x_train.shape:\n', x_train.shape)
print('y_train.shape:\n', y_train.shape)

print('x_test.shape:\n', x_test.shape)
print('y_test.shape:\n', y_test.shape)