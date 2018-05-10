__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import math
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sklearn import utils as skutils
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class DnnModel( object ):
    
    def __init__( self, dataset, gpu, log_path ):
        self._dataset       = dataset
        self._batch_size    = 100
        self._data_pos      = 0

        self._filter_size   = 3
        self._feature_map   = 32
        self._pool_size     = 2
        self._weight_decay  = 0.001

        self._train_steps   = 10000
        self._min_lr        = 0.0004
        self._max_lr        = 0.002
        self._decay_speed   = 3000

        self._gpu = gpu
        self._log_path  = log_path
        self._print_interval    = 300

        self._is_training    = tf.placeholder( dtype = tf.bool )
        self._learning_rate  = tf.placeholder( dtype = tf.float32 )
        self._X = tf.placeholder( dtype = tf.float32, shape = [ None, self._dataset.cfg._row, self._dataset.cfg._column, 1 ] )
        self._Y = tf.placeholder( dtype = tf.float32, shape = [ None, self._dataset.cfg._class_num] )

    def one_hot( self, y, n_values ):
        return np.eye( n_values )[ np.array( y, dtype = np.int32 ) ]

    def load_data( self ):
        train_x, train_y, test_x, test_y = self._dataset.load_data()

        assert train_x.shape[0] == train_y.shape[0]
        assert test_x.shape[0]  == test_y.shape[0]

        train_x = np.reshape( train_x,  [ -1, self._dataset.cfg._row, self._dataset.cfg._column, 1 ] )
        test_x  = np.reshape( test_x,   [ -1, self._dataset.cfg._row, self._dataset.cfg._column, 1 ] )

        self._train_x, self._train_y = skutils.shuffle(train_x, self.one_hot( train_y, self._dataset.cfg._class_num ), random_state=0)
        self._test_x, self._test_y = skutils.shuffle(test_x, self.one_hot( test_y, self._dataset.cfg._class_num ), random_state=0)

        print( "data loaded!" )
        print(  "train_x shape:{}".format( self._train_x.shape ) + \
                "train_y shape:{}".format( self._train_y.shape ) + \
                "test_x shape:{}".format( self._test_x.shape ) + \
                "test_y shape:{}".format( self._test_y.shape ) )

    def next_batch( self ):

        train_size = self._train_x.shape[0]
        scale = self._data_pos + self._batch_size

        if scale > train_size:
            a = scale - train_size
            x1 = self._train_x[ self._data_pos: ]
            x2 = self._train_x[ 0: a ]
            y1 = self._train_y[ self._data_pos: ]
            y2 = self._train_y[ 0: a ]

            self._data_pos = a
            return np.concatenate( ( x1, x2 ) ), np.concatenate( ( y1, y2 ) )
        else:
            x = self._train_x[ self._data_pos: scale ]
            y = self._train_y[ self._data_pos: scale ]

            self._data_pos = scale
            return x, y


    def build_model( self ):

        batch_norm_params = {
            'decay': 0.995,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ]
        }

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn = tf.nn.relu,
                            normalizer_fn = slim.batch_norm,
                            normalizer_params = batch_norm_params,
                            weights_initializer = slim.initializers.xavier_initializer(),
                            weights_regularizer = slim.l2_regularizer( self._weight_decay ) ):
            
            with slim.arg_scope( [ slim.batch_norm, slim.dropout ], is_training = self._is_training ):
                with slim.arg_scope( [ slim.conv2d, slim.max_pool2d ], stride = 1, padding = 'VALID'):

                    net = slim.conv2d( self._X, self._feature_map, [ self._filter_size, self._filter_size ], scope = 'conv2d_1_3x3' ) # 38x83x32
                    net = slim.conv2d( net,     self._feature_map, [ self._filter_size, self._filter_size ], scope = 'conv2d_2_3x3' ) # 36x36x32

                    net = slim.max_pool2d( net, kernel_size = 2, stride = 2, scope = 'pool2d_3_2x2'  ) # 18x18x32

                    net = slim.conv2d( net,     self._feature_map, [ self._filter_size, self._filter_size ], scope = 'conv2d_4_3x3' ) # 16x16x32
                    net = slim.conv2d( net,     self._feature_map, [ self._filter_size, self._filter_size ], scope = 'conv2d_5_3x3' ) # 14x14x32

                    net = slim.max_pool2d( net, kernel_size = 2, stride = 2, scope = 'pool2d_6_2x2'  ) # 7x7x32

                    net = slim.flatten( net,    scope = 'flaten_7' )

                    net = slim.fully_connected( net, 256, scope = 'fc_8_256' )
                    
                    predictions = slim.fully_connected( net, self._dataset.cfg._class_num, activation_fn = None, scope = 'fc_output' )

        classification_loss = slim.losses.softmax_cross_entropy( predictions, self._Y )
        total_loss      = slim.losses.get_total_loss( add_regularization_losses = True )
        train_step      = tf.train.AdamOptimizer( self._learning_rate ).minimize( total_loss )
        
        correct_preds   = tf.equal( tf.arg_max( self._Y, 1 ), tf.arg_max( predictions, 1 ) )
        accuracy        = tf.reduce_mean( tf.cast( correct_preds, tf.float32 ) )

        tf.summary.scalar( "classification loss", classification_loss )
        tf.summary.scalar( "total loss", total_loss )
        tf.summary.scalar( "accuracy", accuracy )
        tf.summary.scalar( "learning", self._learning_rate )

        merged = tf.summary.merge_all()
        update_ops  = tf.get_collection( tf.GraphKeys.UPDATE_OPS ) # add batch_norm update node

        self._predictions   = predictions
        self._train_step    = train_step
        self._merged        = merged
        self._update_ops    = update_ops


    def run_model( self ):

        os.environ["CUDA_VISIBLE_DEVICES"] = str( self._gpu ) # gpu selection

        sess_config = tf.ConfigProto()  
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 40% gpu
        sess_config.gpu_options.allow_growth = True      # dynamic growth

        with tf.Session( config = sess_config ) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            train_writer    = tf.summary.FileWriter( self._log_path + '/train', graph = tf.get_default_graph() )
            best_accuracy   = 0.
            best_f1_score   = 0.

            for i in range( self._train_steps ):
                x, y    = self.next_batch( )
                lr      = self._min_lr + ( self._max_lr - self._min_lr ) * math.exp( -i / self._decay_speed )

                summary, _, _ = sess.run( [ self._merged, self._update_ops, self._train_step ], feed_dict={
                    self._X: x,
                    self._Y: y,
                    self._learning_rate:    lr,
                    self._is_training:      True
                } )

                train_writer.add_summary( summary, i )

                if i % self._print_interval == 0 :
                    labels  = np.empty( [0] )
                    preds   = np.empty( [0] )

                    for start, end in zip(  range( 0, self._test_x.shape[0], self._batch_size ),
                                            range( self._batch_size, self._test_x.shape[0], self._batch_size ) ):
                        
                        preds_batch = sess.run( self._predictions, feed_dict = { 
                            self._X: self._test_x[ start: end ],
                            self._Y: self._test_y[ start: end ],
                            self._is_training: False
                        } )

                        # preds   = np.append( ( preds,     np.argmax( preds_batch, 1 ) ), axis = 1 )

                        labels  = np.concatenate( [ labels, np.argmax( self._test_y[ start: end ], 1 ) ] )
                        preds   = np.concatenate( [ preds,  np.argmax( preds_batch, 1 ) ] )

                    accuracy    = accuracy_score( labels, preds, range( self._dataset.cfg._class_num ) )
                    f1          = f1_score( labels, preds, range( self._dataset.cfg._class_num ), average = 'macro' )
                    conf_matrix     = confusion_matrix( labels, preds, range( self._dataset.cfg._class_num ) )

                    best_accuracy = max( best_accuracy, accuracy )
                    best_f1_score = max( best_f1_score, f1 )

                    print( "step {:>6}  accuracy: {:>10}  f1_score: {:>10}".format( i, accuracy, f1 ) )
                    print( conf_matrix )
            
            print( "best_accuracy: {}  best_f1_score: {}".format( best_accuracy, best_f1_score ) )

        print( "finish!" )