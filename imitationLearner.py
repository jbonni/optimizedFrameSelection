import tensorflow as tf
import time
import param
import dataHandler as dh
import numpy as np
from sklearn.neighbors import LSHForest
import h5py

import NeuralNetwork as nn

tf.logging.set_verbosity(0)

class imitationLearner():
    def __init__(self, session, param = param.neuralParam):
        #learner session
        self.sess = session
        #parameters of the learner
        self.param = param
        #placeholders
        self.input_ph = tf.placeholder(
                tf.float32, self.param.flat_input_shape, name="inputData_ph")
        self.output_ph = tf.placeholder(
                tf.float32, self.param.output_shape, name="output_ph")
        self.label_ph = tf.placeholder(
                tf.float32, self.param.output_shape, name="label_ph")
        self.keep_prob = tf.placeholder(
                tf.float32, name="dropout_ph")

        #Creates the learner and every operation necessary
        self.output_ph = nn.createLearner(self.input_ph,
                                        self.keep_prob,
                                        self.param)
        self.loss_op = self.loss()
        self.train_op = self.training()
        self.eval_op = self.evaluation()

        #datahandler for data manipulation
        self.dh = dh.dataHandler()


    def loss(self):
#        loss_op = tf.losses.log_loss(self.output_ph,self.label_ph)
        if self.param.loss == "mse":
            loss_op = tf.reduce_sum(tf.square(self.output_ph - self.label_ph))
#        elif self.param.loss == "cos":
#            loss_op = tf.losses.cosine_distance(self.label_ph,
#                                                self.output_ph)
#        elif self.param.loss == "hinge":
#            loss_op = tf.losses.hinge_loss(self.label_ph,
#                                           self.output_ph)
        elif self.param.loss == "softmax":
            loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                                                      labels=tf.nn.softmax(self.label_ph),
                                                      logits=tf.nn.softmax(self.output_ph)))
#        elif self.param.
        else:
            print('loss function not defined')
            raise
#
##        tf.losses.log_loss(self.output_ph,
##                                     self.label_ph,
##                                     weights=2.0,
##                                     epsilon = 0.01)

#        loss_op = tf.nn.softmax_cross_entropy_with_logits(
#                                                      labels=tf.nn.softmax(self.label_ph),
#                                                      logits=self.output_ph)
        return loss_op



    def training(self):
        # Add a scalar summary for the snapshot loss.
#        tf.summary.scalar('loss', tf.reduce_mean(self.loss_op))
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(self.param.learning_rate)
        # Create a variable to track the global step.
#        train_op = tf.train.AdamOptimizer(self.param.learning_rate).minimize(self.loss_op)

#        train_op = tf.train.RMSPropOptimizer(self.param.learning_rate).minimize(self.loss_op)
        global_step = tf.Variable(0, name='global_step', trainable=False)
#        # Use the optimizer to apply the gradients that minimize the loss
#        #(and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(self.loss_op, global_step=global_step)
        return train_op



    def evaluation(self):
        correct_prediction = tf.equal(tf.argmax(self.label_ph,1),
                                      tf.argmax(self.output_ph,1))
        #Cast to floating point numbers and then take the mean
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def run(self):
        #Loads the MNIST dataset
        print("run")
        self.dh.setFilename("mnist.hdf5")#--------------modify
        testBatch = self.dh.loadBatch("test")
        lsh = LSHForest(min_hash_match=8, n_candidates=10, n_estimators=50,
          n_neighbors=10, radius=10.0, radius_cutoff_ratio=0.5,
          random_state=42)
        dhGen = dh.dataHandler()

        total = self.dh.loadRange(0,70000)
#        print(total)
        total = total[:][0:794]
        lsh.fit(total)

        for i, data in enumerate(total):
            distances, indices =  lsh.kneighbors(data[50], n_neighbors=2)
    #        print(indices)
    #        print(distances)
            if distances[0,0] < 1.1e-15:
                pass#identical in mem

            if distances[0,1] < 2.0e-01:
                pass#close in mem
            else:
                dhGen.addData(data)

        raise

        #separate data
        dataShape = self.dh.getDataShape()
        testData = testBatch[:,dataShape[0,0]:dataShape[0,1]]
        testLabel = testBatch[:,dataShape[1,0]:dataShape[1,1]]
#        self.dh.setFilename("rl_data.hdf5")
#        with h5py.File(self.dh.fileName, "r",  libver='latest') as f:
#            print("openFile")
#            testList = f["batch/test/list"]
#            testBatch = self.dh.load(testList[0:2000])
#            dataShape = self.dh.getDataShape()
#            testData = testBatch[:,dataShape[0,0]:dataShape[0,1]]
#            testLabel = testBatch[:,dataShape[1,0]:dataShape[1,1]]

#        dhGen = dh.dataHandler()
#        dataShape = self.dh.getDataShape()
#        print(dataShape)
#        raise

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        #start the session
        self.sess.run(init)

        #load the test batch
        print(" ".join(["loading",self.dh.fileName]))
#        testBatch = self.dh.loadBatch("test")


        #Training loop
        print("Begin training")
#        start_time = time.time()
        for step in range(self.param.max_steps):
            batch = self.dh.loadBatch()


            feed_dict = {self.input_ph:batch[:,dataShape[0,0]:dataShape[0,1]],
                         self.label_ph:batch[:,dataShape[1,0]:dataShape[1,1]],
                         self.keep_prob: 0.5}


            # Run one step of the model.
            _, loss = self.sess.run([self.train_op,self.loss_op],
                                    feed_dict=feed_dict)






#
#            if step >= 30000:
#                for d, l in zip(batch[:,dataShape[0,0]:dataShape[0,1]],loss):
#                    dhGen.addData(d, l)

            #Show loss of network
#            if step % 10 == 0:
#              duration = time.time() - start_time
#              print('Step {0}: loss = {1:.10f} ({2:.3f} sec)'.format(step, loss, duration))
#              start_time = time.time()


            #Evaluate the model periodically.
            if (step + 1) % 100 == 0 or (step + 1) == self.param.max_steps:
                print("%g"%self.sess.run(self.eval_op,
                                                  feed_dict={
                                            self.input_ph: testData,
                                            self.label_ph: testLabel,
                                            self.keep_prob: 1.0}))
#                start_time = time.time()
#        dhGen.saveData()
#        tr, tst = dhGen.randList(64)
#        dhGen.createBatch(tr,"training")
##        dhGen.createBatch(tst,"test")
#        return dhGen



if __name__ == "__main__":
    sess = tf.Session()
    imit = imitationLearner(sess)
    d = imit.run()
