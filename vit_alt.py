from config import *


import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import time


import tensorflow as tf
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, Precision, Recall





CONFIGURATION = CONFIGURATION

class Patch_encoder(Layer):
    def __init__(self,HIDDEN_SIZE,N_PATCHES):
        super(Patch_encoder , self).__init__(name = "patch_encoder")
        self.linear_projection = Dense(HIDDEN_SIZE)
        self.positional_embedding = Embedding(N_PATCHES , HIDDEN_SIZE)
        self.N_PATCHES = N_PATCHES

        
    def call(self , x):
        x = tf.image.resize(x , [256,256])
        patches = tf.image.extract_patches(
            images=x,
            sizes =[1 ,  CONFIGURATION["PATCH_SIZE"] ,CONFIGURATION["PATCH_SIZE"] , 1 ],
            strides = [1 ,  CONFIGURATION["PATCH_SIZE"] ,CONFIGURATION["PATCH_SIZE"] , 1 ],
            rates= [1,1,1,1],
            padding = "VALID"
            
        )
        
        patches_dim = patches.shape[-1]
        patches = tf.reshape(patches , (tf.shape(x)[0] , -1 , patches_dim))
        
        embedding_input = tf.range(start=0, limit=self.N_PATCHES, delta=1)
        output = self.linear_projection(patches) + self.positional_embedding(embedding_input)

        return output
    
    
class Transformer_encoder(Layer):
    def __init__(self , N_HEAD , HIDDEN_SIZE):
        super(Transformer_encoder , self).__init__(name = "transformer_encoder")
        self.normalization_layer1 = LayerNormalization()
        self.normalization_layer2 = LayerNormalization()
        
        self.attention_layer = MultiHeadAttention(num_heads = N_HEAD , key_dim = HIDDEN_SIZE )
        
        self.Dense_layer_1 = Dense(HIDDEN_SIZE , activation  = tf.nn.gelu)
        self.Dense_layer_2 = Dense(HIDDEN_SIZE , activation  = tf.nn.gelu)
        
    
    def call(self , input):
        x1 = self.normalization_layer1(input)
        x1 = self.attention_layer(x1 , x1)
        x1 = Add()([x1 , input])
        x2 = self.normalization_layer2(x1)
        x2 = self.Dense_layer_1(x2)
        output = self.Dense_layer_2(x2)
        output = Add()([output , x1])
        
        return output
    
    
class VIT(Model):
    def __init__(self , N_HEAD , HIDDEN_SIZE , N_PATCHES ,    N_LAYERS , N_DENSE_UNITS ):
        super(VIT , self).__init__(name = "visual_transformer")

        self.N_LAYERS = N_LAYERS
        
        self.patch_encoder_layer = Patch_encoder(HIDDEN_SIZE,N_PATCHES)
        self.transformer_encoder_layer1 = [Transformer_encoder(N_HEAD , HIDDEN_SIZE) for _ in range(int(N_LAYERS*0.5))]
        self.transformer_encoder_layer2 = [Transformer_encoder(N_HEAD , HIDDEN_SIZE) for _ in range(int(N_LAYERS*0.5))]
        
        self.Dense_layer_1 = Dense(N_DENSE_UNITS , activation = tf.nn.gelu)
        self.Dense_layer_2 = Dense(N_DENSE_UNITS , activation = tf.nn.gelu)
        self.Dense_layer_3 = Dense(CONFIGURATION["NUM_CLASSES"] , activation = "softmax")


    def call(self , inputs):
        
        x = self.patch_encoder_layer(inputs)
        x1 = x
        x2 = x
        for i in range(int((self.N_LAYERS)*0.5)):
            x1 = self.transformer_encoder_layer1[i](x1)
        for i in range(int((self.N_LAYERS)*0.5)):
            x2 = self.transformer_encoder_layer2[i](x2)
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x = Add()([x1,x2])
        x = self.Dense_layer_1(x)
        x = self.Dense_layer_2(x)        
        return  self.Dense_layer_3(x)
    
    
  
        


    

def load_metrics():
    loss_function = CategoricalCrossentropy()

    metrics = [
        CategoricalAccuracy(name="accuracy"),
        TopKCategoricalAccuracy(k=2, name="top_k_accuracy"),
        Precision(name="precision"),
        Recall(name="recall")
    ]
    
    return metrics , loss_function
    
def build_vit():
    vit = VIT(N_HEAD = 4, HIDDEN_SIZE = 768, N_PATCHES = 256,
    N_LAYERS = 8, N_DENSE_UNITS = 128)
    vit(tf.zeros([32,121,121,3]))
    print(vit.summary())
    metrics , loss_function  = load_metrics()
    vit.compile(optimizer = Adam(learning_rate = CONFIGURATION["LEARNING_RATE"] ),
           loss = loss_function,
           metrics = metrics)
    
    return vit


def load_dataset():
    train_directory = "test_train_dataset/train"
    test_directory = "test_train_dataset/test"

    val_directory = "test_train_dataset/val"


    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CONFIGURATION["CLASS_NAMES"],
        color_mode='rgb',
        batch_size=CONFIGURATION["BATCH_SIZE"],
        image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        shuffle=True,
        seed=99,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CONFIGURATION["CLASS_NAMES"],
        color_mode='rgb',
        batch_size=1,#CONFIGURATION["BATCH_SIZE"],
        image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        shuffle=True,
        seed=99,
    )


    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CONFIGURATION["CLASS_NAMES"],
        color_mode='rgb',
        batch_size=1,#CONFIGURATION["BATCH_SIZE"],
        image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
        shuffle=True,
        seed=99,
    )
    augment_layers = tf.keras.Sequential([
    RandomRotation(factor = (-0.025, 0.025)),
    RandomFlip(mode='horizontal',),
    RandomContrast(factor=0.1),
    ])
    def augment_layer(image, label):
        return augment_layers(image, training = True), label
    
    
    training_dataset = (
        train_dataset
        .map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    testing_dataset = (
        test_dataset
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = (
        val_dataset
        .prefetch(tf.data.AUTOTUNE)
    )
    return training_dataset , testing_dataset , validation_dataset




def plot_history(history):
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Validation Accuracy')
    plt.legend()

    # Plot loss vs val_loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Validation Loss')
    plt.legend()

    plt.show()


def evaluate(model , test_dataset):
    predictions = model.predict(test_dataset)
    predicted_classes = np.argmax(predictions, axis=1)

    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    true_classes = np.argmax(true_labels, axis=1)

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:\n", conf_matrix)

    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    print("F1 Score: {:.4f}".format(f1))


    return

def training_vit(vit , training_dataset , val_dataset):
    start_time = time.time()

    history = vit.fit(
        training_dataset,
        validation_data = val_dataset,
        epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1, batch_size = CONFIGURATION['BATCH_SIZE'])

    end_time = time.time()
    
    time_taken = end_time - start_time
    
    return history , time_taken


def main():
    
    vit = build_vit()
    training_dataset , test_dataset , val_dataset = load_dataset()
    history , time_taken = training_vit(vit , training_dataset , val_dataset)

    print(f"Time taken for training: {time_taken:.2f} seconds")
    
    plot_history(history)
    
    evaluate(vit , test_dataset)
        
    return None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
########################################################################################################################







if __name__ == "__main__":
    CONFIGURATION = CONFIGURATION

    main()

