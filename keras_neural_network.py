import tensorflow as tf
from tensorflow import keras
import numpy as np

class keras_neural_network(object):
    def __init__(self, n_classes=4, batch_size=64, nvariables=8, learning_rate=0.0001):
    
        self.n_classes     = n_classes
        self.batch_size    = batch_size
        self.nvariables    = nvariables
        self.learning_rate = learning_rate

        self.model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(self.nvariables,)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(self.n_classes, activation=tf.keras.activations.linear) ])

        #optimizer = tf.train.RMSPropOptimizer(0.001)
        #optimizer = tf.train.AdamOptimizer(0.0005)
        #self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        #self.model.compile(loss='huber_loss', optimizer=optimizer, metrics=['huber_loss'])
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))

    def train_neural_network(self,dataX,dataY,EPOCHS):
        self.model.fit(dataX, dataY, epochs=EPOCHS, verbose=0)

    def test_neural_network(self,dataX):
        predY = self.model.predict(dataX)#.flatten()
        return predY

    def copy_weights(self,neural_net_weights):
        self.model.set_weights(neural_net_weights)
        
    def save_model(self,filename):
        keras.models.save_model(self.model, filename, overwrite=True, include_optimizer=True)

    def get_weights(self):
        return self.model.get_weights()

#    def load_model(self,filename):
#        loaded_model = keras.models.load_model(filename, custom_objects=None, compile=True)
#        return loaded_model

