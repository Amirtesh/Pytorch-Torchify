# Pytorch-TorchKit
Custom library for pytorch neural networks when working with image data or tabular data(supervised learning) with keras like compile and fit methods.

Contains features like learning rate scheduler,gradient clipping, plotting losses and accuracy(in classification) and r2_score(for regression),metrics method with model to get accuracy,f1,precision for classification and mean squared error,root mean squared error,mean absolute error,r2_score for regression to evaluate on a validation dataset easily without having to predict and get required metrics manually.

Usage:

1.While creating your model class just inherit from TorchKit.ImageClassifier.ImageModel when working with image data or from TorchKit.TabularData.TabularModel when working with tabular data(supervised learning). When working with tabular data once you have made the train and test splits make sure to create a train and test TensorDataset and dataloader with a batch size of your size.  
2.Create an instance of your model.  
3.Use compile method to pass in loss function,optimizer,learning_rate_scheduler,gradient clip. When using TabularModel an additional parameters called task is present which is by default set to 'classification'.  
4.Use fit method to pass in number of epochs,train and validation loaders. This is common for both ImageModel and TabularModel. If you want to visualize the loss,accuracy and r2_score plots, make sure to create a variable of your choice to which fit method is assigned(For example, history=model.fit(...)). Note that loss plot is common for both TabulatModel and ImageModel but accuracy plot of validation loader is only for ImageModel and TabularModel when task is set to 'classification'. r2_score plot is present only in TabularModel when task is 'regression'.  
5.Use predict method to make predictions either on a tensor,a dataset or a dataloader.  
6.Use metrics method by passing in a dataset. This return accuracy,f1 and precision of model's performance on the dataset in case of ImageModel and in TabularModel when task is 'classfication'(order of variables returned- accuracy,f1_score,precision). In TabularModel with task as 'regression' this method returns mean squared error,root mean squared error,mean absolute error and r2_score in order.  
