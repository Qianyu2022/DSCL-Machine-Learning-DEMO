'''
Transfer Learning DEMO
@ Authour: DSCL at Uconn
Nov, 17th, 2022
'''

'''
Python: 3.8.7
Tensorflow version: 2.3.0
matplotlib version: 3.6.2
'''
# Be careful with the version of the packages and install them priorly
import tensorflow as tf
import matplotlib.pyplot as plt

# Store the image dataset in your drive and modify the file path below
image_folder ='D:/qiz19014/OneDrive - University of Connecticut/Anomaly Detection CNN/Data/Train_Images'

# Loading the training images and preprocessing the images for uniform image size
image_size = (224,224)
batch_size = 4
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_folder,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset="training",
    seed=1364,
    image_size=image_size,
    batch_size=batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_folder,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset="validation",
    seed=1364,
    image_size=image_size,
    batch_size=batch_size,
)

# You check by this code if you have loaded the all the images and how many categories your dataset is comprised of
train_classes = train_ds.class_names
test_classes = test_ds.class_names
print(train_classes)
print("The number of batches: ", len(train_ds))



# Start building the model and define the parameters

# Input data size
img_shape = (224, 224, 3)

# The model we used in transfer learning is VGG16 which has been integrated in the packages
# And the parameters and knowledge we transfer were obtained from Imagenet.

VGG16_features = tf.keras.applications.VGG16(input_shape=img_shape,
                                             include_top = False,
                                             weights='imagenet')
VGG16_features.summary()

# Low level layers of VGG16 are frozen in the training process
VGG16_features.trainable=False

# But we need to add some high level layers for the target dataset
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(train_classes),activation='softmax')
fc1 = tf.keras.layers.Dense(64, activation='relu')


# Add the new layers to the frozen layers
model = tf.keras.Sequential([
    VGG16_features,
    global_average_layer,
    fc1,
    prediction_layer
])
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(),
              #Fine tune with the learning rate
              #optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

# It has been predefined when loading the training dataset as the batch size is set as 4
# Then the step number has to be total number/batch size
steps_per_epoch = len(train_ds)
validation_steps=len(test_ds)


# Epochs means how many times you want the model to look through the entire dataset completely
Epochs = 15

# All the training metric you want to know is stored in the history and can be recalled by the keywords
history = model.fit(train_ds,
                    epochs=Epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    validation_data=test_ds)

# Test performance
loss_test, accuracy_test=model.evaluate(test_ds,steps = validation_steps)
print("loss:{:.2f}".format(loss_test))
print("accuracy: {:.2f}".format(accuracy_test))

# Accuracy history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








