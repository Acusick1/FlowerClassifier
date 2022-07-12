# Appsilon assignment

---

## Process

At a high level, an overview of the steps taken in this assignment is as follows:
1. Explored dataset and separated into classes (30 min)
2. Selected classifier method 
3. Defined data preprocessing (30 min)
4. Experimented with model architecture (60 min)
5. Implemented saving and evaluate logic (30 min)
6. Implemented simple hyperparameter tuning (60 min)
7. Refactoring/bug fixing (60 min)
8. Analysing results/report writing (120 min)

Total time spent: 6.5 hours

The problem seeks to classify the species of flower from a given image, so the dataset was separated into subdirectories
containing images of each class. The dataset is fairly clean, with most images containing a centred flower or selection
of single species flowers. It is also balanced, with each class represented by 80 images. The images themselves are of 
mixed orientation (landscape and portrait) but are of high quality, generally having a minimum dimension of 500 pixels.

A convolutional neural network (CNN) was chosen as the classifier. This was an obvious choice as CNN's learn features 
within the image throughout the training process (as opposed to manual time-consuming feature engineering), and ability
to process images with considerably fewer parameters than regular neural networks.

Rather than attempting to load all images into memory for a given training run, a file dataset was used to store the 
image path only, along with a preprocessing function that enables images and labels to be loaded on-demand. TensorFlow 
dataset performance optimisation tools such as caching and prefetching were used to limit the processor load and avoid 
idle GPU time while a new batch is loaded into memory.

A number of architectures were experimented with in terms of image resizing and number of convolutional layers. The 
chosen method is a standard approach in fully convolutional neural networks: halving the image size after each "same"
padded convolution layer and doubling the number of filters. This allows increasingly complex features to be identified
as the image passes through the network. 

An image size of (256, 256, 3) was chosen as the input size to the network, balancing image quality with computational 
resources. Four convolutional + max pooling layers are used to extract features from the image, followed by a dense
layer and finally an output layer. Dropout was used between these two final layers to reduce the chance of over-fitting
the data.

A simple tuning method was implemented for some key hyperparameters, randomly selecting a new configuration from the
defined search space and refitting the optimal configuration on the full training dataset. The implementation of this 
tuning had a desirable effect on model accuracy as will be discussed below.

## Model evaluation

Before training a model, the dataset was split into train (90%) and test (10%) sets. Currently this split is carried out
at runtime, however with additional time the on-disk dataset should be separated to produce a well-defined and evenly
distributed test set. The train set was further split to provide a validation set (20%). The Adam optimiser was used 
with a sparse categorical cross entropy function, which minimises the difference between the predicted class 
probabilities and the truth distribution (a one-hot vector), and allows a single integer based output layer to be used.

Early stopping was used with respect to the validation loss to terminate training and restore the model to the best 
observed state if it did not improve within a set number of iterations. Validation loss was also used as the metric 
to determine the best model during hyperparameter tuning.

The model performs well on the test set, scoring ~85% during a single training run and >95% when tuned. This increase 
is to be expected since the best model found during tuning is retrained on the full training dataset (train + 
validation). Further improvements could be made by testing additional architectures and including more hyperparameters
in an optimisation based model tuning.

Following training the output model is evaluated on the test and full dataset, showing accuracy and confusion matrices 
for both. From inspection of the confusion matrices, it can be seen the model confuses for colts' foot for daffodils
(5 images out of 80) and cowslips for tulips (3 out of 80), due to their similar colours and shapes. Both tulips and 
irises are difficult to classify generally, with each being confused for 5 separate classes on at least one occasion.
This is logical as the tulip changes shape significantly throughout flowering, whereas the iris comes in a variety of
different colours.

## Questions
### How would you share your findings with the client?
In presenting the findings of this feasibility study to the client I would show a simple breakdown of the model and
the key metrics such as accuracy on unseen data (test set) and the time taken to predict the species of flower from an
input image using the trained model. Furthermore, I would visualise the results using standard methods such as confusion
matrix, and point to any issues in class distinguishing that require further attention and/or data.

### What would your comments be to a colleague building the app, regarding the model?
To a colleague looking to leverage the model I would explain its interface in terms of inputs and outputs. This would 
cover what the model itself expects as an input, how to input image files using the functions created, and the
mapping between output integer values and their associated classes. I would also recommend that the model be hosted 
centrally and message brokers used to transfer inputs/outputs in a robust manner.