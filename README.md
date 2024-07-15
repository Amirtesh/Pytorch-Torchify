# Pytorch-TorchKit
Custom library for pytorch neural networks when working with image data or tabular data(supervised learning) with keras like compile and fit methods.

Contains features like learning rate scheduler,gradient clipping, plotting losses and accuracy(in classification) and r2_score(for regression),metrics method with model to get accuracy,f1,precision for classification and mean squared error,root mean squared error,mean absolute error,r2_score for regression to evaluate on a validation dataset easily without having to predict and get required metrics manually.

Usage:

1.While creating your model class just inherit from TorchKit.ImageClassifier.ImageModel when working with image data or from TorchKit.TabularData.TabularModel when working with tabular data(supervised learning)  
2.
