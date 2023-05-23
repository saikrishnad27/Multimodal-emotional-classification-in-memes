import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv1D, Embedding, GlobalAveragePooling1D 
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
df = pd.read_csv('memotiondataset from kaggle')
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df = df.drop(columns = ['text_ocr', 'overall_sentiment'])
df.head()
df = df.replace({'humour': {'not_funny': 0, 'funny': 1, 'very_funny': 2, 'hilarious':3},
            'sarcasm': {'not_sarcastic': 0, 'general': 1, 'twisted_meaning': 2, 'very_twisted': 3},
            'offensive': {'not_offensive': 0, 'slight': 1, 'very_offensive': 2, 'hateful_offensive': 3},
            'motivational': {'not_motivational': 0, 'motivational': 1}})
cleaned = df.copy()
cleaned.dropna(inplace=True)
cleaned.isnull().any()
width = 100
height = 100
X = []
for i in tqdm(range(cleaned.shape[0])):
    if i in [119, 4799, 6781, 6784, 6786]:
        pass
    else:
        path = '../input/memotion-dataset-7k/memotion_dataset_7k/images/'+cleaned['image_name'][i]
        img = image.load_img(path,target_size=(width,height,3))
        img = image.img_to_array(img)
        img = img/255.0
        X.append(img)
        
X = np.array(X)
target = cleaned.iloc[:,2:]
target.head()
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.2)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomContrast([.5,2]),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(X)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
 base_model_1 =tf.keras.applications.resnet.ResNet152(input_shape=X[0].shape,
                                               include_top=False,
                                               weights='imagenet')
base_model_2 = tf.keras.applications.VGG19(input_shape=X[0].shape,
                                               include_top=False,
                                               weights='imagenet')
base_model_1.trainable = False
base_model_2.trainable = False
def image_model():
    image_input = tf.keras.Input(shape=(150, 150, 3), name = 'image_input')
    image_layers = data_augmentation(image_input)
    image_layers = preprocess_input(image_layers)
    layer_bm_1 = base_model_1(image_input, training=False)
    dropout_layer = Dropout(0.2)(layer_bm_1)
    layer_bm_1 = Conv2D(2048, kernel_size=2,padding='valid')(layer_bm_1)
    dropout_layer = Dropout(0.3)(layer_bm_1)
    layer_bm_1 = Dense(512)(dropout_layer)
    dropout_layer = Dropout(0.5)(layer_bm_1)
    layer_bm_2 = base_model_2(image_input, training=False)
    dropout_layer = Dropout(0.4)(layer_bm_2)
    layer_bm_2 = Dense(512)(layer_bm_2)
    dropout_layer = Dropout(0.2)(layer_bm_2)
    layers = tf.keras.layers.concatenate([layer_bm_1, layer_bm_2])
    dropout_layer = Dropout(0.3)(layers)
    image_layers = GlobalAveragePooling2D()(layers)
    image_layers = Dropout(0.5, name = 'dropout_layer')(image_layers)
    return image_input, image_layers
image_input, image_layers = image_model()
def standardization(data):
    data = data.apply(lambda x: x.lower())
    data = data.apply(lambda x: re.sub(r'\d+', '', x))
    data = data.apply(lambda x: re.sub(r'\w*.com\w*', '', x, flags=re.MULTILINE))
    data = data.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return data

cleaned['text_corrected'] = standardization(cleaned.text_corrected)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
vocab_size = 100000
sequence_length = 100

vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

text_ds = np.asarray(cleaned['text_corrected'])
vectorize_layer.adapt(tf.convert_to_tensor(text_ds))
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(cleaned.text_corrected, target, test_size = 0.2)
embedding_dim=32

def text_model():
    text_input = tf.keras.Input(shape=(None,), dtype=tf.string, name='text')
    text_layers = vectorize_layer(text_input)
    text_layers = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(text_layers)
    dropout_layer = Dropout(0.3)(text_layers)
    
    text_layers = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, activation='relu', return_sequences=True))(text_layers)
    dropout_layer = Dropout(0.4)(text_layers)
    text_layers = tf.keras.layers.BatchNormalization()(text_layers)

    text_layers = tf.keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(text_layers)
    dropout_layer = Dropout(0.2)(text_layers)
    text_layers = tf.keras.layers.GlobalMaxPooling1D()(text_layers)
    dropout_layer = Dropout(0.5)(text_layers)
    
    text_layers = tf.keras.layers.Dense(2048, activation="relu")(text_layers)
    text_layers = tf.keras.layers.Dropout(0.2)(text_layers)
    return text_input, text_layers

text_input, text_layers = text_model()
def model(layer_1, layer_2, image_input, text_input):
    concatenate = tf.keras.layers.concatenate([layer_1, layer_2], axis=1)
    semi_final_layer = tf.keras.layers.Dense(2048, activation='relu')(concatenate)

    prediction_layer_1 = tf.keras.layers.Dense(4, activation='softmax', name = 'humuor')
    prediction_layer_2 = tf.keras.layers.Dense(4, activation='softmax', name = 'sarcasm')
    prediction_layer_3 = tf.keras.layers.Dense(4, activation='softmax', name = 'offensive')
    prediction_layer_4 = tf.keras.layers.Dense(2, activation='softmax', name = 'motivational')

    output_1 = prediction_layer_1(semi_final_layer)
    output_2 = prediction_layer_2(semi_final_layer)
    output_3 = prediction_layer_3(semi_final_layer)
    output_4 = prediction_layer_4(semi_final_layer)

    model = tf.keras.Model(inputs = [image_input, text_input] , 
                           outputs = [output_1, output_2, output_3, output_4])
    return model
model = model(image_layers, text_layers, image_input, text_input)
import os
# Define the checkpoint directory to store the checkpoints
checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
def decay(epoch):
  if epoch < 5:
    return 1e-3
  elif epoch >= 5 and epoch < 15:
    return 1e-4
  else:
    return 1e-5
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]
base_learning_rate = 0.0001
losses = {
      "humuor": "sparse_categorical_crossentropy", 
      "sarcasm": "sparse_categorical_crossentropy", 
      "offensive": "sparse_categorical_crossentropy", 
      "motivational": "sparse_categorical_crossentropy"
}
lossWeights = {
      "humuor": 1.0, 
      "sarcasm": 1.0, 
      "offensive": 1.0, 
      "motivational": 1.0
}
metrics = {
      "humuor": "sparse_categorical_accuracy", 
      "sarcasm": "sparse_categorical_accuracy", 
      "offensive": "sparse_categorical_accuracy", 
      "motivational": "sparse_categorical_accuracy"
}
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss = losses,
              loss_weights= lossWeights,
              metrics=metrics)
tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
history = model.fit(x = {"image_input": X_train, "text_input": X_text_train},
                    y = {"sarcasm": y_train.sarcasm, 
                         "humuor": y_train.humour, 
                         "offensive": y_train.offensive, 
                         "motivational": y_train.motivational},
                    batch_size=32,
                    epochs=30,
                    validation_data=({"image_input": X_test, "text_input": X_text_test}, 
                                     {"sarcasm": y_test.sarcasm, 
                                      "humuor": y_test.humour, 
                                      "offensive": y_test.offensive, 
                                      "motivational": y_test.motivational}),
                    callbacks=callbacks
                   )
  evaluate = model.evaluate(x = {"image_input": X_test, "text_input": X_text_test},
                    y = {"sarcasm": y_test.sarcasm, 
                         "humuor": y_test.humour, 
                         "offensive": y_test.offensive, 
                         "motivational": y_test.motivational},
                    batch_size=32,
                   )
  predictions = model.predict(x = {"image_input": X_test, "text_input": X_text_test})
