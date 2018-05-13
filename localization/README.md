# course
CV coursework

Localization in CUBdataset using vgg16
The example in cv slide uses ResNet18 for fine tuning, but I didn't find the ResNet18 checkpoint file on the net, so choose vgg16

Dataset: CUBdataset
    
    Provided by the teaching assistant
    80% for training, 20% for testing

Model: vgg16( the last layer has 4 output( x, y, w, h ) )

    fine-tuning in vgg16, the checkpoint file was downloaded from http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    In addition to the last layer, other network parameters are restored from the checkpoint file

Run instruction: python dnn_model.py -v version -g gpu

    version:    any string     for differentiate different processes and logs
    gpu:    0 ~ n (n : the number of gpu available on the computer)  for specify gpu used for compute( only used in dnn model )

    eg: "python dnn_model.py -v 000 -g 0"


Tips:

    1. use tensorboard to watch the log and compute graph( usage tensorboard --logdir=dir --port=600x )