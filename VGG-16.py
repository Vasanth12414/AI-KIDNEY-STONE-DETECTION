!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download safurahajiheidari/kidney-stone-images
import zipfile
zip_ref = zipfile.ZipFile('/content/kidney-stone-images.zip')
zip_ref.extractall('/content')
zip_ref.close()


DATADIR = '/content/KidneyDisease'
CATEGORIES  = ['TestImages','TrainImages','ValidImages']

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

DATADIR = '/content/KidneyDisease'
CATEGORIES  = ['TestImages','TrainImages','ValidImages']

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    images = os.listdir(path)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle(f'{category}', fontsize=18)

    for i in range(5):
        img_name = images[np.random.randint(0, len(images))]
        img_path = os.path.join(path, img_name)
        img_array = cv2.imread(img_path)

        axes[i].imshow(img_array)
        axes[i].axis('off')

    plt.show()
	img_array.shape
	IMG_SIZE = 224
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array)
training_data = []
def create_train_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    # images = os.listdir(path)
    labels = CATEGORIES.index(category)
    for img in os.listdir(path):
      try:
        img_path = os.path.join(path,img)
        img_array = cv2.imread(img_path)
        new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        training_data.append([new_array,labels])
      except Exception as e:
        pass
create_train_data()


x = []
y = []
for features , labels in training_data:
  x.append(features)
  y.append(labels)
  
  x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

print(f'x_train Length: {x_train.shape[0]},X_train Image Size : {x_train.shape[1:3]},X_train Channel Dimension : {x_train.shape[3]}')
print(f'x_test Length: {x_test.shape[0]},X_test Image Size : {x_test.shape[1:3]},X_test Channel Dimension : {x_test.shape[3]}')

import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg16

vgg = vgg16.VGG16(weights = 'imagenet',include_top = False,input_shape = (IMG_SIZE,IMG_SIZE,3))

for layer in vgg.layers:
  layer.trainable = False
  
  model = keras.Sequential([
    vgg,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1024,activation = 'relu'),
    keras.layers.Dense(512,activation = 'relu'),
    keras.layers.Dense(3,activation = 'softmax')

])
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
			  
			  model.fit(x_train,y_train,epochs = 20)
			  
			  loss , accuracy = model.evaluate(x_test,y_test)
print(f'Model Accuracy: {accuracy * 100}')

pred = np.argmax(model.predict(x_test), axis = -1)

pred

y_test[:5]

from sklearn.metrics import classification_report , confusion_matrix
print(classification_report(y_test,pred))

cf = confusion_matrix(y_test,pred , normalize='true')
import seaborn as sns
sns.heatmap(cf , annot=True,cmap='crest')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()