import tensorflow as tf 
import numpy as np 

SEED = 23455

rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand()/10.0 - 0.05)] for (x1, x2) in x]
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape :
        y = tf.matmul(x, w1)
        loss_mse = tf.reduce_mean(tf.square(y_ - y))

    grads = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print('After %d training steps, w1 is: ' %(epoch))
        print(w1.numpy(), '\n')

print('Final w1 is:', w1.numpy())