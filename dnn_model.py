__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sklearn import utils as skutils
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class DnnModel( object ):
    
    def __init__( self, dataset, log_path, version, gpu ):
        self._dataset       = dataset
        self._batch_size    = 50
        self._data_pos      = 0
        self._gpu           = gpu
        self._log_path      = log_path
        self._version       = version

        self._filter_size   = 3
        self._feature_map   = 32
        self._pool_size     = 2
        self._weight_decay  = 0.001

        self.is_training    = tf.placeholder( dtype = tf.bool )
        self.learning_rate  = tf.placeholder( dtype = tf.float32 )
        self.X = tf.placeholder( dtype = tf.float32, shape = [ None, self._dataset._row, self._dataset._column, 1 ] )
        self.Y = tf.placeholder( dtype = tf.float32, shape = [ None, self._dataset._class_num] )


    def one_hot( self, y, n_values ):
        return np.eye( n_values )[ np.array( y, dtype = np.int32 ) ]


    def load_data( self ):
        train_x, train_y, test_x, test_y = self._dataset.load_data()

        train_x = np.reshape( train_x, [ -1, self._dataset._row, self._dataset._column, 1 ] )
        train_y = self.one_hot( train_y, self._dataset._class_num )

        test_x = np.reshape( test_x,  [ -1, self._dataset._row, self._dataset._column, 1 ] )
        test_y = self.one_hot( test_y, self._dataset._class_num )

        self._train_x, self._train_y = skutils.shuffle(train_x, train_y, random_state=0)
        self._test_x, self._test_y = skutils.shuffle(test_x, test_y, random_state=0)

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
            x1 = self._train_x[ self._train_x: ]
            x2 = self._train_x[ 0: a ]
            y1 = self._train_y[ self._train_x: ]
            y2 = self._train_y[ 0: a ]

            self._data_pos = a
            return np.concatenate( ( x1, x2 ) ), np.concatenate( ( y1, y2 ) )
        else:
            x = self._train_x[ self._data_pos: scale ]
            y = self._train_y[ self._data_pos: scale ]

            self._data_pos = scale
            return x, y

    def build_model( self, inputs, labels ):

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
            
            with slim.arg_scope( [ slim.batch_norm, slim.dropout ], is_training = self.is_training ):
                with slim.arg_scope( [ slim.conv2d, slim.max_pool2d ], stride = 1, padding = 'VAILD'):

                net = slim.conv2d( inputs,  self._feature_map, self._filter_size, scope = 'conv2d_1_3x3' ) # 38x83x32
                net = slim.conv2d( net,     self._feature_map, self._filter_size, scope = 'conv2d_2_3x3' ) # 36x36x32

                net = slim.max_pool2d( net, kernel_size = 2, stride = 2, scope = 'pool2d_3_2x2'  ) # 18x18x32

                net = slim.conv2d( net,     self._feature_map, self._filter_size, scope = 'conv2d_4_3x3' ) # 16x16x32
                net = slim.conv2d( net,     self._feature_map, self._filter_size, scope = 'conv2d_5_3x3' ) # 14x14x32

                net = slim.max_pool2d( net, kernel_size = 2, stride = 2, scope = 'pool2d_6_2x2'  ) # 7x7x32

                net = slim.flatten( net,    scope = 'flaten_7' )

                net = slim.fully_connected( net, 256, scope = 'fc_8_256' )
                
                preds = slim.fully_connected( net, self._dataset._class_num, activation_fn = None, scope = 'fc_output' )
        
        classification_loss = slim.losses.softmax_cross_entropy( preds, labels )
        total_loss  = slim.losses.get_total_loss( add_regularization_losses = True )
        optimizer   = tf.train.AdamOptimizer( self.learning_rate )
        train_op    = slim.learning.create_train_op( total_loss, optimizer )

        self._train_op = train_op



    def run_model( self ):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str( self._gpu )
        tf_cfg  = tf.ConfigProto( )
        tf_cfg.gpu_options.allow_growth = True

        with tf.Session( config = tf_cfg ) as sess:

            slim.learning.train( 
                self._train_op,
                logdir              = self._log_path,
                number_of_steps     = 1000,
                save_summaries_secs = 300,
                save_interval_secs  = 1200
             )