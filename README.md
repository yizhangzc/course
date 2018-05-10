# course
CV coursework

Classification in AffNIST dataset( including svm_model, knn_model, dnn_model )

Dataset: AffNIST from https://www.cs.toronto.edu/~tijmen/affNIST/
    
    transformed/training_and_validation_batches/* used for training
    
    transformed/test.mat used for testing


Run instruction: python main.py -m model -v version -g gpu

    model: svm / dnn / knn     for use different clasification model
    version:    any string     for differentiate different processes and logs
    gpu:    0 ~ n (n : the number of gpu available on the computer)  for specify gpu used for compute( only used in dnn model )

    eg: "python main.py -m svm -v 000 -g 0"


Tips:

    1. use tensorboard to watch the log and compute graph( usage tensorboard --logdir=dir --port=600x )
    2. for speeding up the experiment, only a part of the data is used. if you want use all data of the dataset, Remove the slicing operation( eg: data["affNISTdata"][0][0][5][0][0: 1000] -> data["affNISTdata"][0][0][5][0] )