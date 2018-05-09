__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 14.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

import svm_model
import knn_model
import dnn_model
import data
import argparse

def main( model ):
    dataset = data.AffNIST()

    if model == "svm" :
        Model = svm_model.SvmModel( dataset )
    elif model == "dnn":
        Model = dnn_model.DnnModel( dataset )
    else:
        Model = knn_model.KnnModel( dataset )

    # Model = knn_model.KnnModel( DataSet )
    Model.load_data()
    Model.training()
    Model.evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="model selection for AffNIST classification")
    parser.add_argument(    '-m', '--model', type=str.lower, help='Type of using model', 
                            default="svm", choices = ["svm", "knn", "dnn"], required=False)
    args = parser.parse_args()

    main( args.model )