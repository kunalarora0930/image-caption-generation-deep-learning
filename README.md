# Image Caption Generator

colab notebook: https://colab.research.google.com/drive/1Wj6iXXIHhu_gS1qaHF9t9w6VmNlUwczu#scrollTo=YH6qeyddsgIi
link to dataset used: https://www.kaggle.com/datasets/adityajn105/flickr8k/

## Overview

Generating image captions using deep learning. The images are taken from *Flickr8k* dataset from Kaggle. The system employs an encoder-decoder neural network architecture to generate descriptive captions for images.

## Neural Network Architecture

Certainly! The neural network architecture used in the given code is an image captioning model that consists of an Encoder-Decoder architecture. Let's break down the components of the neural network:

### Encoder:
The encoder is responsible for extracting meaningful features from input images. In this code, the encoder utilizes the pre-trained VGG16 (Visual Geometry Group 16-layer) model, which is a deep convolutional neural network (CNN) commonly used for image classification tasks.

```python
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
```

- **VGG16 Model:** The VGG16 model is loaded with pre-trained weights. It consists of 16 layers, including convolutional and fully connected layers. The model is designed for image classification and has been trained on large image datasets.

- **Output Modification:** The last layer (classification layer) is removed, and the model is modified to output the activations of the second-to-last layer (`model.layers[-2].output`). This layer is a dense layer with 4096 neurons, which serves as a high-level representation of the input image.

### Decoder:
The decoder generates descriptive captions based on the features extracted by the encoder. It processes the information from the image features and generates a sequence of words that form the caption.

```python
# Sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.2)(se1)
se3 = LSTM(256)(se2)
```

- **Embedding Layer:** The tokenized captions are passed through an embedding layer. This layer converts each token into a dense vector of fixed size (256 in this case). It helps in representing words in a continuous vector space.

- **Dropout Layer:** A dropout layer is added to prevent overfitting. It randomly sets a fraction of input units to zero during training.

- **LSTM (Long Short-Term Memory) Layer:** The embedded sequence is then processed by an LSTM layer. LSTM is a type of recurrent neural network (RNN) that is capable of capturing long-term dependencies in sequential data. The LSTM layer outputs a fixed-size vector (256 dimensions in this case) representing the context of the input sequence.

### Decoder-Overall Model Integration:

```python
# Decoder Model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

- **Combining Encoder and Decoder:** The output of the encoder (image features) and the output of the decoder (sequence features) are combined using the `add` layer. This merged representation is then processed through additional dense layers.

- **Dense Layers:** Two dense layers are added to refine the combined features. The first dense layer has 256 units with the ReLU activation function, and the second dense layer outputs a probability distribution over the vocabulary using the softmax activation function. This distribution represents the likelihood of each word in the vocabulary being the next word in the caption.

- **Model Compilation:** The overall model is compiled with categorical cross-entropy loss and the Adam optimizer in preparation for training.

This neural network architecture forms an end-to-end image captioning model where the encoder extracts image features, and the decoder generates captions based on these features. The model is trained using the provided dataset and is later used to generate captions for new images.
