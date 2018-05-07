__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5

import data
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

class SvmModel( object ):

    def __init__( dataset ):
        
        self.dataset = dataset


    def load_data( self ):
        
        train_x, train_y, test_x, test_y, Dcfg = self.dataset.load_data()


    def build_model():

        classifier = svm.