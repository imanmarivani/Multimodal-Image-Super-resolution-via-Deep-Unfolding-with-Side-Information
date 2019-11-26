import os
import cv2
import numpy as np
import tensorflow as tf
import utils
import time

class DMSC():

    def __init__(self, sess, config):
        self.config = config
        self.sess = sess
        self.batch_size = config['batch_size']
        self.train_dir = config['train_dir']
        self.D_size = config['dictionary_size']
        self.SC_size = config['sparse_code_size']
        self.input_patch_size = config['input_patch_size']
        self.out_patch_size = config['out_patch_size']
        self.patch_feature = config['patch_feature']
        self.learning_rate = config['learning_rate']
        self.tetta_init = config['tetta_init']
        self.epoch = config['epoch']
        self.model()
        tf.global_variables_initializer().run()

    def model(self):

        self.LR_image = tf.placeholder(tf.float32, [None,None,None,1], name = "LR_image")
        self.HR_image = tf.placeholder(tf.float32, [None,None,None,1], name = "HR_image")
        self.HR_sideinfo = tf.placeholder(tf.float32, [None,None,None,1], name = "HR_sideinfo")

        self.weights = {
		        'wH' : tf .Variable(tf.random_normal([self.input_patch_size,self.input_patch_size,1,self.patch_feature], stddev = .1, dtype = tf.float32), name = 'wH'),
		        'wd' : tf.Variable(tf.random_normal([self.patch_feature,self.SC_size], stddev = .1, dtype = tf.float32), name = 'wd'),
		        's' : tf.Variable(tf.random_normal([self.SC_size,self.SC_size], stddev = .1, dtype = tf.float32), name = 's'),
		        'Dx' : tf.Variable(tf.random_normal([self.SC_size,self.D_size], stddev = .1, dtype = tf.float32), name = 'Dx'),
		        'wG' : tf.Variable(tf.random_normal([self.input_patch_size,self.input_patch_size,self.D_size,1], stddev = .1, dtype = tf.float32), name = 'wG'),
                'tetta1' : tf.Variable(self.tetta_init, name = 'tetta1'),
                'tetta2' : tf.Variable(self.tetta_init, name = 'tetta2'),
                   
                'wH2' : tf .Variable(tf.random_normal([self.input_patch_size,self.input_patch_size,1,self.patch_feature], stddev = .1, dtype = tf.float32), name = 'wH2'),
		        'wd2' : tf.Variable(tf.random_normal([self.patch_feature,self.SC_size], stddev = .1, dtype = tf.float32), name = 'wd2'),
		        's2' : tf.Variable(tf.random_normal([self.SC_size,self.SC_size], stddev = .1, dtype = tf.float32), name = 's2'),
		        'Dx2' : tf.Variable(tf.random_normal([self.SC_size,self.D_size], stddev = .1, dtype = tf.float32), name = 'Dx2'),
		        'wG2' : tf.Variable(tf.random_normal([self.out_patch_size,self.out_patch_size,self.D_size,1], stddev = .1, dtype = tf.float32), name = 'wG2'),
                'tetta12' : tf.Variable(self.tetta_init, name = 'tetta12'),
                'tetta22' : tf.Variable(self.tetta_init, name = 'tetta22'),
           
                'wH1' : tf .Variable(tf.random_normal([self.input_patch_size,self.input_patch_size,1,self.patch_feature], stddev = .1, dtype = tf.float32), name = 'wH1'),
		        'wd1' : tf.Variable(tf.random_normal([self.patch_feature,self.SC_size], stddev = .1, dtype = tf.float32), name = 'wd1'),
		        's1' : tf.Variable(tf.random_normal([self.SC_size,self.SC_size], stddev = .1, dtype = tf.float32), name = 's1'),
                'tetta11' : tf.Variable(self.tetta_init, name = 'tetta11'),
                'tetta21' : tf.Variable(self.tetta_init, name = 'tetta21')
		        }
      
        
        self.pred = self.network()
        self.loss = tf.reduce_mean(tf.square(self.HR_image - self.pred))

        self.saver = tf.train.Saver()

    def train_model(self):
        print("**** training .... *****")
		
        # LR_image_set, HR_image_set, side_info = data_prep.read_data(self.train_dir)

        self.train_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        start_time = time.time()

        for ep in range(self.epoch):
                
            ERR = 0
            c = 0

            batch_idx = len(HR_image_set)//self.batch_size
            for idx in range(batch_idx):

                batch_LR_images = LR_image_set[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_HR_images = HR_image_set[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_HR_sideinfo = side_info[idx*self.batch_size:(idx+1)*self.batch_size]

                _, error = self.sess.run([self.train_optimizer, self.loss], feed_dict = {self.LR_image: batch_LR_images, self.HR_image: batch_HR_images, self.HR_sideinfo: batch_HR_sideinfo})

            
        self.saver.save(self.sess, './modelfinal/modelfinal')
        

    def load_model(self,saved_dir):
        print("***Loading models***")
        self.saver.restore(self.sess,tf.train.latest_checkpoint(saved_dir))
        g2 = tf.get_default_graph()

    def predict(self,imgl,imgh):
        print("***Super-resolving the test image***")
        self.predict = self.network()
        g = self.sess.run(self.predict, feed_dict = {self.LR_image: imgl, self.HR_sideinfo: imgh})
        return g[0,:,:,0]
        
        
    def network(self):
        eta = utils.proxLeSITA #LeSITA proximal operator for l1-l1 minimization 

        #Convolutional layer as patch extractor 
        convfea1 = tf.nn.conv2d(self.HR_sideinfo, self.weights['wH1'], strides = [1,1,1,1], padding = 'SAME')
        size_convfea1 = tf.shape(convfea1)
        convfea1 = tf.reshape(convfea1, [-1, size_convfea1[1]*size_convfea1[2],self.patch_feature])

        #LISTA sparse coding for the side information branch
        wd1 = tf.reshape(tf.matmul(tf.reshape(convfea1, [-1, self.patch_feature]), self.weights['wd1']), [-1,size_convfea1[1]*size_convfea1[2], self.SC_size])
        z01 = utils.ShLU(wd1,self.weights['tetta11'])
        ss1 = tf.reshape(tf.matmul(tf.reshape(z01, [-1, self.SC_size]), self.weights['s1']), [-1,size_convfea1[1]*size_convfea1[2], self.SC_size])
        z = utils.ShLU(ss1+wd1,self.weights['tetta21'])
    
        convfea = tf.nn.conv2d(self.LR_image, self.weights['wH'], strides = [1,1,1,1], padding = 'SAME')
        size_convfea = tf.shape(convfea)
        convfea = tf.reshape(convfea, [-1, size_convfea[1]*size_convfea[2],self.patch_feature])
        wd = tf.reshape(tf.matmul(tf.reshape(convfea, [-1, self.patch_feature]), self.weights['wd']), [-1,size_convfea[1]*size_convfea[2], self.SC_size])
        z0 = eta(wd,z,self.weights['tetta1'])
        ss = tf.reshape(tf.matmul(tf.reshape(z0, [-1, self.SC_size]), self.weights['s']), [-1,size_convfea[1]*size_convfea[2], self.SC_size])
        U = eta(ss+wd,z,self.weights['tetta2'])
        hpatch = tf.reshape(tf.matmul(tf.reshape(U, [-1, self.SC_size]), self.weights['Dx']), [-1,size_convfea[1],size_convfea[2] , 64])
        out = tf.nn.conv2d(hpatch, self.weights['wG'], strides = [1,1,1,1], padding = 'SAME')
        
        convfea2 = tf.nn.conv2d(out, self.weights['wH2'], strides = [1,1,1,1], padding = 'SAME')
        size_convfea2 = tf.shape(convfea2)
        convfea2 = tf.reshape(convfea2, [-1, size_convfea2[1]*size_convfea2[2],self.patch_feature])
        wd2 = tf.reshape(tf.matmul(tf.reshape(convfea2, [-1, self.patch_feature]), self.weights['wd2']), [-1,size_convfea2[1]*size_convfea2[2], self.SC_size])
        z02 = utils.ShLU(wd2,self.weights['tetta12'])
        ss2 = tf.reshape(tf.matmul(tf.reshape(z02, [-1, self.SC_size]), self.weights['s2']), [-1,size_convfea2[1]*size_convfea2[2], self.SC_size])
        U2 = utils.ShLU(ss2+wd2,self.weights['tetta22'])
        hpatch2 = tf.reshape(tf.matmul(tf.reshape(U2, [-1, self.SC_size]), self.weights['Dx2']), [-1,size_convfea2[1],size_convfea2[2] , 64])
        #Convolutional layer as patch aggregator
        out = tf.nn.conv2d(hpatch2, self.weights['wG2'], strides = [1,1,1,1], padding = 'SAME')

        return out
 
