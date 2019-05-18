#*** Logistic Regression ***

# What I'm trying to do now is to figure out wheter the number is prime or not
# But something interesting, it doesn't work!!!

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2

from numpy import reshape
from others import isPrime, isEven
from sklearn.model_selection import train_test_split
import tensorflow as tf

Xt = [i for i in range(1, 100)]
Yt = [[0, 1] if isPrime(i) else [1, 0] for i in Xt]
x_train = reshape(Xt,(-1,1))
y_train = reshape(Yt, (-1, 2))
x_train, x_test, y_train, y_test = train_test_split(x_train, 
                                                    y_train, 
                                                    test_size = 0.33, 
                                                    random_state = 1)
print("{0}, {1}, {2}, {3}".format(len(x_train), len(y_train), len(x_test), len(y_test)))

model = Sequential()
model.add(Dense(2,  # output dim is 2, one score per each class
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                input_dim=1))  # input dimension = number of features your data has
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

def Prediction(n):
    p = model.predict(x=[[n]])
    return [0 if i < 0.5 else 1 for i in p[0]]

for n in range(1, 30):
    print("The number {0} is {1} prime".format(n, Prediction(n)))

# Using Tensorflow

X = reshape([i for i in range(1, 100)], (-1, 1))
Y = reshape([isPrime(i) for i in Xt], (-1, 1))

W = {'oculta': tf.Variable(tf.random_normal([1, 3]), name='w_oculta'),
     'saida': tf.Variable(tf.random_normal([3, 1]), name='w_saida')}
b = {'oculta': tf.Variable(tf.random_normal([3]), name='b_oculta'),
     'saida': tf.Variable(tf.random_normal([1]), name='b_saida')}

xph = tf.placeholder(tf.float32, [99, 1], name='xph')
yph = tf.placeholder(tf.float32, [99, 1], name='yph')

oculta = tf.add(tf.matmul(xph, W['oculta']), b['oculta'])
ativacao_oculta = tf.sigmoid(oculta)
saida = tf.add(tf.matmul(ativacao_oculta, W['saida']), b['saida'])
ativacao_saida = tf.sigmoid(saida)

err = tf.losses.mean_squared_error(yph, ativacao_saida)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(err)

init = tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)
    for epocas in range(1000):
        mean_erro = 0
        _, custo = s.run([opt, err], feed_dict={xph: X, yph: Y})
        if epocas % 20 == 0:
            print(custo)
    wf, bf = s.run([W, b])

oculta = tf.add(tf.matmul(xph, wf['oculta']), bf['oculta'])
ativacao_oculta = tf.sigmoid(oculta)
saida = tf.add(tf.matmul(ativacao_oculta, wf['saida']), bf['saida'])
ativacao_saida = tf.sigmoid(saida)

with tf.Session() as s:
    s.run(init)
    print(s.run(ativacao_saida, feed_dict={xph: X}))