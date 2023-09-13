Craeting AlexNet on CIFAR10 dataset

AlexNet was the first convolutional network which used GPU to boost performance. 

1. AlexNet architecture consists of 5 convolutional layers, 3 max-pooling layers, 2 normalization layers, 2 fully connected layers, and 1 softmax layer. 

2. Each convolutional layer consists of convolutional filters and a nonlinear activation function ReLU. 

3. The pooling layers are used to perform max pooling. 

4. Input size is fixed due to the presence of fully connected layers.

5. The input size is mentioned at most of the places as 224x224x3 but due to some padding which happens it works out to be 227x227x3 

6. AlexNet overall has 60 million parameters.

Model Details 
The model which won the competition was tuned with specific details-

1. ReLU is an activation function 

2. Used Normalization layers which are not common anymore 

3. Batch size of 128 

4. SGD Momentum as learning algorithm 

5. Heavy Data Augmentation with things like flipping, jittering, cropping, color normalization, etc. 

6. Ensembling of models to get the best results. 
