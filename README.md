# Perceptron with Softmax

This is an implementation of Perceptron using Softmax for classification. This model is built from scratch, so it may not be optimal and has some bugs (which I am not aware of).

## Requirements
1. &nbsp;NumPy

## Functions
<div>
  <b> 1. Train model: </b>
  To train the Perceptron model, call the fit() function. the fit() function accepts 4 arguments:
  
  > 1. x for the features
  > 2. y for the target class
  > 3. lr for the learning rate value 
  > 4. epoch for the number of epoch.
</div>

<div>
  <b> 2. Predict: </b>
  Predicting data can be executed by calling the predict() function, which takes 1 input: list of features. The output is a list of prediction
</div>

<div>
  <b> 3. Accuracy: </b>
  Count the accuracy of list of prediction and list of actual class by calling the accuracy() function. Takes 2 input:

  > 1. yPredicted for the list of prediction
  > 2. yActual for the list of actual target
</div>
<hr>