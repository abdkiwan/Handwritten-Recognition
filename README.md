# Handwritten Recognition using Convolutional Neual Network
Training a convolutional neural network for handwritten digits on the famous MNIST dataset.


## Dependencies
This code is written in python. To use it you will need:
* python  2.7
* Keras  2.2.2
* numpy  1.15.1
* Tensorflow or Theano 


## Loading Data
    from mnist_loader import load_dataset
    train_x, train_y, test_x, test_y = load_dataset()


## Training
    from ConvolutionalNN import CNN
    cnn = CNN()
    cnn.train(train_x, train_y, epochs=10, batch_size=32)


## Evaluation
    score = cnn.evaluate(test_x, test_y)
    print('Loss is : ', score[0])
    print('Accuracy is : ', score[1])    


## Classify a new instance
    new_class = cnn.classify(np.array([test_x[0]]))


## Save and Load the trained model
    cnn.save_model('saved_model')
    cnn.load_model('saved_model')
