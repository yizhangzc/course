__author__ = 'yizhangzc'

# course:       CV
# teacher:      DongHui Wang
# author:       zju_cs / Yi Zhang / 21721190
# mail:         yizhangzc@gmail.com
# data:         2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import numpy as np

import scipy.io as sio

class DatasetConfig( object ):

    def __init__( self ):
        self._dataset_path =  "/data/zy/course/cv/"
        self._class_num = 10
        self._row = 40
        self._column = 40

class AffNIST( object ):

    cfg = DatasetConfig()

    def __init__( self ):
        
        self.cfg._dataset_path = "/data/zy/course/cv/"
        self.cfg._class_num = 10
        self.cfg._row = 40
        self.cfg._column = 40

    def load_data( self ):

        print( "loading data..." )

        train_data_path = self.cfg._dataset_path + "training_and_validation_batches/"
        test_data_path  = self.cfg._dataset_path

        train_x = np.empty( [ 0, 1600 ], dtype = np.float32 )
        train_y = np.empty( [ 0 ], dtype = np.int32 )

        for i in range( 32 ):

            data = sio.loadmat( train_data_path + "{}.mat".format( i + 1 ) )
            train_x = np.vstack( ( train_x, np.transpose( data["affNISTdata"][0][0][2] )[0: 200] ) )
            train_y = np.concatenate( [ train_y, data["affNISTdata"][0][0][5][0][0: 200] ] )
            print( "{}.mat finished!".format( i + 1 ) )


        test_data = sio.loadmat( test_data_path + "test.mat" )
        print( "test.mat finished!" )


        test_x = np.transpose( test_data["affNISTdata"][0][0][2] )[0: 200]
        test_y = test_data["affNISTdata"][0][0][5][0][0: 200]

        test_x.astype( np.float32 )
        test_y.astype( np.int32 )

        train_x = np.true_divide( train_x,  255. )
        test_x  = np.true_divide( test_x,   255. )

        print(  "train_x shape:{} type:{}\n".format( train_x.shape, train_x.dtype ) + \
                "train_y shape:{} type:{}\n".format( train_y.shape, train_y.dtype ) + \
                "test_x shape:{} type:{}\n".format( test_x.shape, test_x.dtype ) + \
                "test_y shape:{} type:{}".format( test_y.shape, test_y.dtype ) )
        
        print( "data loaded!" )

        return train_x, train_y, test_x, test_y