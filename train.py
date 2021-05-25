
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np


# Initializing the CNN
classifier = tf.keras.models.Sequential()
# First convolution layer and pooling
classifier.add(tf.keras.layers.Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))

# Pooling
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(tf.keras.layers.Flatten())

# Adding a fully connected layer
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=6, activation='softmax')) # softmax for more than 2


classifier.summary()
# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
                  
history=classifier.fit(
        training_set,
        steps_per_epoch=600/5,# No of images in training set
        epochs=200,
        validation_data=test_set,
        validation_steps=30/5)# No of images in test set

#testing the model
X_test,Y_test =[],[]
for ibatch, (X,Y) in enumerate(test_set):
	X_test.append(X)
	Y_test.append(Y)
	ibatch+=1
	if (ibatch== 5*28):break
	
#concatenate everything together
X_test=np.concatenate(X_test)
Y_test=np.concatenate(Y_test)
Y_test=np.int32([np.argmax(r) for r in Y_test])

#get prediction from the model and calculate accuracy
y_pred=np.int32([np.argmax(r) for r in classifier.predict(X_test)])
match=(Y_test==y_pred)
print('Testing accuracy =%.2f%%' % (np.sum(match)*100/match.shape[0]))

#confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import seaborn as sn
plt.figure(figsize=(9,8))
cm=confusion_matrix(Y_test,y_pred)
cm=cm/cm.sum(axis=1)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
sn.heatmap(cm, annot=True ,cmap="YlGnBu")



# Saving the model
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model.h5')


plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
nepochs=len(history.history['loss'])
plt.plot(range(nepochs), history.history['loss'], 'g-', label='train')
plt.plot(range(nepochs), history.history['val_loss'], 'c-', label='test')
plt.legend(prop={'size':20})
plt.ylabel('loss')
plt.xlabel('number of epochs')
plt.subplot(1,2,2)
nepochs=len(history.history['accuracy'])
plt.plot(range(nepochs), history.history['accuracy'], 'g-', label='train')
plt.plot(range(nepochs), history.history['val_accuracy'], 'c-', label='test')
plt.legend(prop={'size':20})
plt.ylabel('accuracy')
plt.xlabel('number of epochs')
plt.show()


