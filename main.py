__author__ = 'yizhangzc'

# course:     CV
# teacher:    DongHui Wang
# author:     zju_cs / Yi Zhang / 21721190
# mail:       yizhangzc@gmail.com
# data:       2018/5
# environment:  ubuntu 16.04 / python 3.5 / numpy 1.14 / tensorflow 1.2 / CUDA 8.0

# import svm_model
import knn_model
# import dnn_model
import data

def main():
    DataSet = data.AffNIST()
    Model = knn_model.KnnModel( DataSet )
    Model.load_data()
    Model.build_model()
    Model.run_model()


if __name__ == '__main__':
    main()