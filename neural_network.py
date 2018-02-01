import tensorflow as tf
import numpy as np

class neural_network(object):

    def __init__(self, hidden1=50, hidden2=200, hidden3=50, n_classes=4, batch_size=100, nvariables=8):

        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.n_classes = n_classes
        self.X = tf.placeholder('float32', shape=(None, nvariables) )
        self.Y = tf.placeholder('float32', shape=(None, n_classes) )

        # Define layers
        self.ly1={'weights':tf.Variable(tf.random_normal([nvariables, self.hidden1])),'biases': tf.Variable(tf.random_normal([self.hidden1]))}
        self.ly2={'weights':tf.Variable(tf.random_normal([self.hidden1,self.hidden2])),'biases': tf.Variable(tf.random_normal([self.hidden2]))}
        self.ly3={'weights':tf.Variable(tf.random_normal([self.hidden2,self.hidden3])),'biases': tf.Variable(tf.random_normal([self.hidden3]))}
        self.out={'weights':tf.Variable(tf.random_normal([self.hidden3,self.n_classes])),\
                  'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        # Build network
        l1 = tf.add(tf.matmul(self.X, self.ly1['weights']), self.ly1['biases'])
        l1 = tf.nn.relu(l1)
        l2 = tf.add(tf.matmul(l1, self.ly2['weights']), self.ly2['biases'])
        l2 = tf.nn.relu(l2)
        l3 = tf.add(tf.matmul(l2, self.ly3['weights']), self.ly3['biases'])
        l3 = tf.nn.relu(l3)
        output = tf.add(tf.matmul(l3, self.out['weights']), self.out['biases'])
        
        # Optimize weights/biases and store values 
        self.output = tf.identity(output)
        self.variables = [self.ly1['weights'],self.ly1['biases'],self.ly2['weights'],self.ly2['biases'],\
                         self.ly3['weights'],self.ly3['biases'],self.out['weights'],self.out['biases']]
        self.saver = tf.train.Saver()

    def train_neural_network(self,sess,dataX,dataY):
        cost = 0.5 * tf.reduce_mean(tf.squared_difference(self.output, self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost) 
	sess.run(self.optimizer, feed_dict = {self.X: dataX, self.Y: dataY})

    def test_neural_network(self,sess,dataX):
        sess.run(tf.global_variables_initializer()) 
        predY = sess.run(self.output, feed_dict = {self.X: dataX})
        return predY

    def copy_weights(self,neural_net):
	to_copy = neural_net.variables
        self.ly1['weights'] = to_copy[0]
        self.ly1['biases']  = to_copy[1]
        self.ly2['weights'] = to_copy[2]
        self.ly2['biases']  = to_copy[3]
        self.ly3['weights'] = to_copy[4]
        self.ly3['biases']  = to_copy[5]
        self.out['weights'] = to_copy[6]
        self.out['biases']  = to_copy[7]

    def save_model(self,sess,filename):
        sess.run(tf.global_variables_initializer())
        self.saver.save(sess,filename)

