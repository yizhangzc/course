# course
CV coursework

Classification in AffNIST dataset( including svm_model, knn_model, dnn_model )

Dataset: AffNIST from https://www.cs.toronto.edu/~tijmen/affNIST/
    
    transformed/training_and_validation_batches/* used for training
    
    transformed/test.mat used for testing


Run instruction: python main.py -m model( model : svm or dnn or knn )

    eg: "python main.py -m svm"