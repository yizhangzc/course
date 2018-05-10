__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import data
from sklearn import utils as skutils
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

class SvmModel( object ):

    def __init__( self, dataset ):
        
        self._dataset   = dataset
        self._kernel    = 'rbf'
        self._gamma     = 'auto'
        self._C         = 1.0


    def load_data( self ):
        
        train_x, train_y, test_x, test_y = self._dataset.load_data()

        self._train_x, self._train_y = skutils.shuffle(train_x, train_y, random_state=0)
        self._test_x, self._test_y = skutils.shuffle(test_x, test_y, random_state=0)


    def build_model( self ):

        self._classifier = svm.SVC( C = self._C, kernel = self._kernel, gamma = self._gamma )
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        self._classifier.fit( self._train_x, self._train_y )

        print( "model built!" )

    def run_model( self ):
        preds = self._classifier.predict( self._test_x )

        accuracy = accuracy_score( self._test_y, preds, range( self._dataset.cfg._class_num ) )
        f1 = f1_score( self._test_y, preds, range( self._dataset.cfg._class_num ) )
        conf_mtrix = confusion_matrix( self._test_y, preds, range( self._dataset.cfg._class_num ) )

        print(  "result: accuracy: {}  f1_score: {}  ".format( accuracy, f1 ) +\
                "confusion_matrix: {}".format( conf_mtrix ) )

        print( "prediction finished!" )