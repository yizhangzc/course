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
    
    def __init__( self, dataset, log_path ):
        self._dataset       = dataset
        self._batch_size    = 50
        self._data_pos      = 0
        self._log_path      = log_path

        self._filter_size   = 3
        self._feature_map   = 32
        self._pool_size     = 2
        self._weight_decay  = 0.001

        self._learning_rate = 0.001


    def load_data( self ):
        train_x, train_y, test_x, test_y = self._dataset.load_data()

        train_x = np.reshape( train_x, [ -1, self._dataset._row, self._dataset._column, 1 ] )
        test_x = np.reshape( test_x,  [ -1, self._dataset._row, self._dataset._column, 1 ] )

        self._train_x, self._train_y = skutils.shuffle(train_x, train_y, random_state=0)
        self._test_x, self._test_y = skutils.shuffle(test_x, test_y, random_state=0)

        print( "data loaded!" )
        print(  "train_x shape:{}".format( self._train_x.shape ) + \
                "train_y shape:{}".format( self._train_y.shape ) + \
                "test_x shape:{}".format( self._test_x.shape ) + \
                "test_y shape:{}".format( self._test_y.shape ) )

    def load_batch( self, samples, labels ):
        x, y = tf.train.batch(
            [ samples, labels ],
            batch_size = self._batch_size,
            allow_smaller_final_batch = True )
        
        return x, y

    def model( self, inputs, is_training ):

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
            
            with slim.arg_scope( [ slim.batch_norm, slim.dropout ], is_training = is_training ):
                with slim.arg_scope( [ slim.conv2d, slim.max_pool2d ], stride = 1, padding = 'VAILD'):

                    net = slim.conv2d( inputs,  self._feature_map, self._filter_size, scope = 'conv2d_1_3x3' ) # 38x83x32
                    net = slim.conv2d( net,     self._feature_map, self._filter_size, scope = 'conv2d_2_3x3' ) # 36x36x32

                    net = slim.max_pool2d( net, kernel_size = 2, stride = 2, scope = 'pool2d_3_2x2'  ) # 18x18x32

                    net = slim.conv2d( net,     self._feature_map, self._filter_size, scope = 'conv2d_4_3x3' ) # 16x16x32
                    net = slim.conv2d( net,     self._feature_map, self._filter_size, scope = 'conv2d_5_3x3' ) # 14x14x32

                    net = slim.max_pool2d( net, kernel_size = 2, stride = 2, scope = 'pool2d_6_2x2'  ) # 7x7x32

                    net = slim.flatten( net,    scope = 'flaten_7' )

                    net = slim.fully_connected( net, 256, scope = 'fc_8_256' )
                    
                    net = slim.fully_connected( net, self._dataset._class_num, activation_fn = None, scope = 'fc_output' )
        
        return net

    def trianing( self ):

        batch_x, batch_y = self.load_batch( self._train_x, self._train_y )

        preds = self.model( batch_x, is_training = True )
        one_hot_labels = slim.one_hot_encoding( batch_y, self._dataset._class_num )

        slim.losses.softmax_cross_entropy( preds, one_hot_labels )
        total_loss  = slim.losses.get_total_loss( add_regularization_losses = True )
        tf.summary.scalar( 'loss', total_loss )

        optimizer   = tf.train.AdamOptimizer( self._learning_rate )
        train_op    = slim.learning.create_train_op( total_loss, optimizer, summarize_gradients = True )

        slim.learning.train( 
            train_op,
            logdir              = self._log_path + "/log/train" ,
            number_of_steps     = 10000,
            save_summaries_secs = 300,
            save_interval_secs  = 1200 )

    def evaluate( self ):
        
        batch_x, batch_y = self.load_batch( self._test_x, self._test_y )
        
        preds = self.model( batch_x, is_training = False )
        preds = tf.to_int64( tf.argmax( preds, 1 ) )

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map( {
            'accuracy': slim.metrics.streaming_accuracy( preds, batch_y ),
            'mse':      slim.metrics.streaming_mean_squared_error( preds, batch_y )
        } )

        summary_ops = []

        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar( metric_name, metric_value )
            op = tf.Print( op, [metric_value], metric_name )
            summary_ops.append( op )

        slim.get_or_create_global_step()

        slim.evaluation.evaluation_loop(
            "",
            self._log_path + "/log/train",
            self._log_path + "/log/test",
            num_evals = 5000,
            eval_op = names_to_updates.values(),
            eval_interval_secs = 60
        )