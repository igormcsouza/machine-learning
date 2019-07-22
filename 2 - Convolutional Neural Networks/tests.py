from convnets import Model
from tensorflow.examples.tutorials import mnist as data

# Deprecated, change way of reading the dataset
mnist = data.input_data.read_data_sets('../databases/mnist/', one_hot=False)
Xtr, Ytr = mnist.train.images, mnist.train.labels
Xte, Yte = mnist.test.images, mnist.test.labels

my_model = Model({
    'Xtr': Xtr, 'Xte': Xte, 'Ytr': Ytr, 'Yte': Yte 
})

my_model.starter()
my_model.train()
print(my_model.evaluate())
print(my_model.predict(Xte[0]))