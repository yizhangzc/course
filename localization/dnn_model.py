__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import argparse
import math
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import data
import vgg
from sklearn import utils as skutils


class DnnModel( object ):
    
    def __init__( self, dataset, gpu, log_path ):
        self._dataset       = dataset
        self._batch_size    = 50 
        self._train_pos     = 0
        self._ckpt_path     = self._dataset._dataset_path + 'vgg_16.ckpt'

        self._train_steps   = 10000

        self._min_lr        = 0.0004
        self._max_lr        = 0.002
        self._decay_speed   = 1000

        self._gpu           = gpu
        self._log_path      = log_path

        self._is_training    = tf.placeholder( dtype = tf.bool )
        self._learning_rate  = tf.placeholder( dtype = tf.float32 )
        self._X = tf.placeholder( dtype = tf.float32, shape = [ None, self._dataset._row, self._dataset._column, 3 ] )
        self._Y = tf.placeholder( dtype = tf.float32, shape = [ None, self._dataset._out_dim] )


    def load_data( self ):

        idxs    = np.arange( self._dataset._img_num )
        random.shuffle( idxs )

        train_idxs  = idxs[ : int( len( idxs ) * 0.8 ) ]
        test_idxs   = idxs[ int( len( idxs ) * 0.8 ):  ]

        self._train_idxs    = train_idxs
        self._test_idxs     = test_idxs

    def next_train_batch( self ):

        train_size = self._train_idxs.shape[0]
        scale = self._train_pos + self._batch_size

        if scale > train_size:
            a = scale - train_size
            x1 = self._train_idxs[ self._train_pos: ]
            x2 = self._train_idxs[ 0: a ]

            self._train_pos = a
            batch_idxs = np.concatenate( ( x1, x2 ) )
        else:
            x = self._train_idxs[ self._train_pos: scale ]

            self._train_pos = scale
            batch_idxs = x
        
        imgs    = np.empty( [0, self._dataset._row, self._dataset._column, 3 ] )
        boxes   = np.empty( [0, self._dataset._out_dim ] )

        for idx in batch_idxs:
            img, box = self._dataset.get_item( idx )
            
            imgs    = np.append( imgs,  np.expand_dims( img, 0 ), axis = 0 )
            boxes   = np.append( boxes, np.expand_dims( box, 0 ), axis = 0 )

        return imgs, boxes


    def build_model( self ):

        predictions, _ = vgg.vgg_16( self._X, num_classes = self._dataset._out_dim, is_training = self._is_training )

        # smooth l1 loss
        localization_loss   = tf.reduce_mean( tf.where( tf.less( tf.abs( self._Y - predictions ), 1. ),
                                            0.5 * tf.square( self._Y - predictions ) ,
                                            tf.abs( self._Y - predictions ) - 0.5 ) )

        slim.losses.add_loss( localization_loss )
        total_loss      = slim.losses.get_total_loss( add_regularization_losses = True )
        
        train_step      = tf.train.AdamOptimizer( self._learning_rate ).minimize( total_loss )

        tf.summary.scalar( "localization loss", localization_loss )
        tf.summary.scalar( "total loss", total_loss )
        tf.summary.scalar( "learning", self._learning_rate )

        merged = tf.summary.merge_all()
        update_ops  = tf.get_collection( tf.GraphKeys.UPDATE_OPS ) # add batch_norm update node

        self._train_step    = train_step
        self._merged        = merged
        self._update_ops    = update_ops


    def run_model( self ):

        os.environ["CUDA_VISIBLE_DEVICES"] = str( self._gpu ) # gpu selection

        sess_config = tf.ConfigProto()  
        sess_config.gpu_options.per_process_gpu_memory_fraction = 1  # 40% gpu
        sess_config.gpu_options.allow_growth = True      # dynamic growth

        # import pdb; pdb.set_trace()
        # variables_to_restore = slim.get_variables_to_restore( exclude = [ 'fc8', 'Adam', 'Adam_1' ] )
        variables_to_restore = slim.get_model_variables()
        variables_to_restore = variables_to_restore[:-2]
        restorer = tf.train.Saver( variables_to_restore )

        with tf.Session( config = sess_config ) as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            restorer.restore( sess, self._ckpt_path )
            print( "model restored!" )

            train_writer    = tf.summary.FileWriter( self._log_path + '/train', graph = tf.get_default_graph() )

            for i in range( self._train_steps ):

                x, y    = self.next_train_batch( )
                lr      = self._min_lr + ( self._max_lr - self._min_lr ) * math.exp( -i / self._decay_speed )

                summary, _, _ = sess.run( [ self._merged, self._update_ops, self._train_step ], feed_dict={
                    self._X: x,
                    self._Y: y,
                    self._learning_rate:    lr,
                    self._is_training:      True
                } )

                train_writer.add_summary( summary, i )


if __name__ == '__main__':
    
    parser  = argparse.ArgumentParser(description="localization in CUBDataset, please chose gpu and version!")
    
    parser.add_argument('-v', '--version',  type=str, default = "", help='model version')
    parser.add_argument('-g', '--gpu',      type=int, default =0,   help='assign task to selected gpu')
    args    = parser.parse_args()

    dataset     = data.CUBDataset()
    log_path    = dataset._dataset_path + 'log/' + args.version + '/'

    Model       = DnnModel( dataset, args.gpu, log_path )

    Model.load_data()
    Model.build_model()
    Model.run_model()
