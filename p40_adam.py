import tensorflow as tf 
from sklearn import datasets
from matplotlib import pyplot as plt 
import numpy as np 
import time

# 搭建第一个神经网络，重要代码，尽量背下来

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target 

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0
######################################
m_w, m_b = 0, 0
v_w, v_b = 0, 0
beta1, beta2 = 0.9, 0.999
delta_w, delta_b = 0, 0
global_step = 0

# 训练部分
now_time = time.time()
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        ####################################################
        global_step += 1
        
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth = 3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        
        grads = tape.gradient(loss, [w1, b1])
        
        ####################################################
        # adam
        m_w = beta1 * m_w + (1 - beta1) * grads[0]
        m_b = beta1 * m_b + (1 - beta1) * grads[1]
        v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
        v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])

        m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
        m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
        v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
        v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))

        w1.assign_sub(lr * m_w_correction/ tf.sqrt(v_w_correction))
        b1.assign_sub(lr * m_b_correction/ tf.sqrt(v_b_correction))


    print('Epoch {}, loss: {}'.format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0 

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis = 1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    
    acc = total_correct / total_number
    test_acc.append(acc)
    print('Test_acc:', acc)
    print('========================================') 
total_time = time.time() - now_time
print('total_time:', total_time)


# 绘制loss曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()
plt.show()

# 绘制Accuracy曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Accuracy$')
plt.legend()
plt.show()

