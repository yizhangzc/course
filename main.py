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

def main( model, log_path ):
    dataset = data.AffNIST()

    if model == "svm" :
        Model = svm_model.SvmModel( dataset, log_path )
    elif model == "dnn":
        Model = dnn_model.DnnModel( dataset, log_path )
    else:
        Model = knn_model.KnnModel( dataset, log_path )

    # Model = knn_model.KnnModel( DataSet )
    Model.load_data()
    Model.training()
    Model.evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="model selection for AffNIST classification")
    parser.add_argument( '-m', '--model', type=str.lower, help='Type of using model',   default="svm", choices = ["svm", "knn", "dnn"], required=False)
    parser.add_argument( '-l', '--log_path', type=str, help='Path to store log files',  default="./", required=True)
    args = parser.parse_args()

    main( args.model, args.log_path )