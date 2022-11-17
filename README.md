### DSCL-Machine-Learning-DEMO
This is a transfer learning demo based on the VGG16 which is being performed on the multi-class classification task.

* **Dataset**

  The dataset zip file contains 40 images for 4 classes.

  Upzip the dataset file and store in your local drive

* **Code Emulation**

 1. Python Configuration
  
  The python code for transfer learning is based on Python 3.8 and Tensorflow 2.x.
  
  You can try first to run the .py file in your python IDE (Pycharm Recommended) with your configuration.
  
  And please check the version of packages installed in your virtual environment if errors are reported.
  
  ```
  import tensorflow as tf
  tf.__version__
  ```
  Python version: 3.8.7
  
  tensorflow version: 2.3.0
  
  matlibplot version: 3.6.2
  
 2. Training
 
 The training will be implemented on CPU by fault if the CUDA is not installed.
 
 
 ## Things you could do with this code:
 
 *a. Fully execute this code with the dataset to check and plot the accuracy and loss*
 
 *b. Modify the hyperparameters for the model selection to figure out the optimal combination*
 
 *c. Replace the dataset with your interested image datasets and repeat the process above*
 
 I will be specifying the parameters you could fine with;
 
 i. Image Size
 
 Corresponding Code:
 
 `image_size = (224,224)` & `image_shape = (224,224,3)`
 
 *Discussion:*
 
 The images involved into the training are generally scaled into the uniform size since the computation capacity cannot afford a bunch of images in high resolution, but meanwhile rescaling would remove some raw information from the original images which might influence the eventual performances, so it would be a trade-off for you to play with.
 
 ii. Learning Rate
 
 ```
 model.compile(optimizer=tf.keras.optimizers.Adam(),
              #Fine tune with the learning rate
              #optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
```

*Discussion:*

Learning rate generally means how much feedback you want the results to return in one batch or how fast you want to model to update the weights in the training process, so increasing the learning rate tends to speed the learning process but there is chance that the optimizer miss the optimal while decreasing the learning rate would render the learning inefficient but might contribute to better performances.


iii. Epochs

`Epochs = 15`

*Discussion:*

I believe the epochs might be intuitive to be understood which determines how many times the model can see the entire datasets. Enough epochs can ensure the model is able to fully extract the critical features from the dataset so that you can observe the loss and accuracy vary as the epochs, but the model would converge after some certain epochs and overmuch eopochs can make the model overfit to the dataset.


---
## Hope you can enjoy playing with the code!
 
