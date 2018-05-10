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

def main( model, version, gpu ):
    dataset = data.AffNIST(  )

    log_path = dataset.cfg._dataset_path + 'log/' + version + '/'

    if model == "svm" :
        Model = svm_model.SvmModel( dataset, gpu, log_path )
    elif model == "dnn":
        Model = dnn_model.DnnModel( dataset, gpu, log_path )
    else:
        Model = knn_model.KnnModel( dataset, gpu, log_path )

    # Model = knn_model.KnnModel( DataSet )
    Model.load_data()
    Model.build_model()
    Model.run_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="model selection for AffNIST classification")
    parser.add_argument('-m', '--model',    type=str.lower, default="svm", help='Type of using model', choices = ["svm", "knn", "dnn"], required=False)
    parser.add_argument('-v', '--version',  type=str, default = "", help='model version')
    parser.add_argument('-g', '--gpu',      type=int, default =0,   help='assign task to selected gpu')
    args = parser.parse_args()

    main( args.model, args.version, args.gpu )