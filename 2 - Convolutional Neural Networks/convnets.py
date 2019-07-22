import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Artificial Inteligence - Convolutional Neural Networks

Hello, Igor here. I was wondering if I can make a lib for Machine Learning an other stuff. 
This is going to be my first try on that. This algoritm have the ability to run Convnets 
using Costum Extimators from Tensor Flow. I'm still figuring out ways of doing this. But 
this seems to wokr so far.

Ways of using...

>>> from convnets import Model

# You have to put all the data on a Dictionary like this:
dataset = {
    'Xtr': ...Trainning data to train
    'Xte': ...Test data to predict
    'Ytr': ...Trainning labels to train
    'Yte': ...Test labels to predict
}

>>> my_model = Model(dataset)

# Model has some atributes, they are all defaulted, unless dataset, they are:
    dataset: You now this one
    frame: Size of the actual image
    kernel_size: Size of the filter, it has to be a tuple (X, X)
    pool_size: Tuple for the pooling 
    strides: How many columns the pooling will be surpress
    units: size of the Dense Network

>>> my_model.starter() # Must be done first, it will create the classifier
>>> my_model.train() # Trainning the model
>>> print(my_model.evaluate()) # Evaluate the actual trainning session
>>> print(my_model.predict(Xte[0])) # Predict one or more data.

'''

class Model():
    def __init__(self, database, frame=(28,28), kernel_size=(5,5), pool_size=(2,2),
        strides=2, units=1024):

        self.Xtr, self.Xte = database['Xtr'], database['Xte']
        self.Ytr = np.asarray(database['Ytr'], dtype=np.int32)
        self.Yte = np.asarray(database['Yte'], dtype=np.int32)
        self.frame = frame
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.units = units

        self.classifier = 0

        print("Model Created Successfully...")

    ''' Function convet
    -> This function is mandatory! So the custume Estimator could work. Also those params are 
    mandatory in this function.

    Features: Means the pixels of the pictures.
    Labels: The output expected
    Mode(optinal): To identify wheter is a training, evaluation or prediction
    '''
    def convnet(self, features, labels, mode):
        _input = tf.reshape(features['X'], [-1, self.frame[0], self.frame[1], 1])
        
        # Get: [batch_size, 28, 28, 1]
        conv1 = tf.layers.conv2d(
            inputs=_input, 
            filters=32,
            kernel_size=self.kernel_size,
            activation=tf.nn.relu,
            padding='same'
        )
        # Give: [batch_size, 28, 28, 32]
        
        # Get: [batch_size, 28, 28, 32]
        pooling1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=self.pool_size, strides=self.strides
        )
        # Give: [batch_size, 14, 14, 32]
        
        # Get: [batch_size, 14, 14, 32]
        conv2 = tf.layers.conv2d(
            inputs=pooling1, 
            filters=64,
            kernel_size=self.kernel_size,
            activation=tf.nn.relu,
            padding='same'
        )
        # Give: [batch_size, 14, 14, 64]
        
        # Get: [batch_size, 14, 14, 64]
        pooling2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=self.pool_size, strides=self.strides
        )
        # Give: [batch_size, 7, 7, 64]
        
        # Get: [batch_size, 7, 7, 64]
        size = pooling2.shape[1] * pooling2.shape[2] * pooling2.shape[3]
        flattening = tf.reshape(pooling2, (-1, size))
        # Give: [batch_size, 3136]
        
        # Get: [batch_size, 3136]
        dense = tf.layers.dense(
            inputs=flattening,
            units=self.units,
            activation=tf.nn.relu
        )
        # Give: [batch_size, 1024]
        
        # This one helps to improve the accuracy, because eliminate a % of the input
        # so the nets could not be overfitted
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.2,
            training=(mode==tf.estimator.ModeKeys.TRAIN)
        )
        
        # Get: [batch_size, 1024]
        _output = tf.layers.dense(
            inputs=dropout,
            units=10
        )
        # Give: [batch_size, 10]
        
        # Get only the grater percentage
        predictions = tf.argmax(_output, axis=1)
        
        # Uses the function as predicted
        if mode==tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )
        
        erro = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=_output)

        if mode==tf.estimator.ModeKeys.TRAIN:
            optmizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train = optmizer.minimize(erro, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=erro,
                train_op=train
            )
        
        if mode==tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {'accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions
            )}
            
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=erro,
                eval_metric_ops=eval_metric_ops
            )

    def starter(self):
        self.classifier = tf.estimator.Estimator(model_fn=self.convnet)
        print("DONE CLASSIFIER!")

    def train(self, batch_size=128, steps=200):
        print("TRAINNING MODEL...")
        train_function = tf.estimator.inputs.numpy_input_fn(
            x={'X': self.Xtr}, 
            y=self.Ytr,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True
        ) #Função de treinamento

        self.classifier.train(input_fn=train_function, steps=steps)
        print("DONE TRAINNING!")
        
    def evaluate(self, epochs=1):
        test_function = tf.estimator.inputs.numpy_input_fn(
            x={'X':self.Xte}, y=self.Yte, num_epochs=epochs, shuffle=False
        )
        results = self.classifier.evaluate(input_fn=test_function)
        
        print("DONE EVALUATING!")
        return results

    def predict(self, sample):
        sample = sample.reshape(1, -1)
        predict_function = tf.estimator.inputs.numpy_input_fn(
            x={'X': sample}, shuffle=False
        )
        predictions = list(self.classifier.predict(input_fn=predict_function))

        return {'classes': predictions}