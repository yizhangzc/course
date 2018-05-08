__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0


import data
from sklearn import utils as skutils
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class KnnModel( object ):

    def __init__( self, dataset ):
        
        self._dataset = dataset
        self._neighbors_num = 5


    def load_data( self ):
        
        train_x, train_y, test_x, test_y = self._dataset.load_data()

        self._train_x, self._train_y = skutils.shuffle(train_x, train_y, random_state=0)
        self._test_x, self._test_y = skutils.shuffle(test_x, test_y, random_state=0)



    def build_model( self ):

        self._classifier = neighbors.KNeighborsClassifier( n_neighbors = self._neighbors_num )
        # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        self._classifier.fit( self._train_x, self._train_y )

        print( "model built!" )


    def run_model( self ):
        preds = self._classifier.predict( self._test_x )

        accuracy = accuracy_score( self._test_y, preds, range( self._dataset._class_num ) )
        f1 = f1_score( self._test_y, preds, range( self._dataset._class_num ) )
        conf_mtrix = confusion_matrix( self._test_y, preds, range( self._dataset._class_num ) )

        print(  "result: accuracy: {}  f1_score: {}  ".format( accuracy, f1 ) +\
                "confusion_matrix: {}".format( conf_mtrix ) )

        print( "prediction finished!" )


