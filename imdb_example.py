"""
This example applies the HAN classifier to Kaggle's IMDB
review dataset. The goal is to predict whether a review is
positive (5 star rating >=3) or negative (otherwise)
"""

import logging
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from keras_han.datasets import TvTropesBookSingleDataset, TvTropesMovieSingleDataset
from keras_han.model import HAN
from keras_han.utils import WeightedBinaryCrossEntropy

# Create a logger to provide info on the state of the
# script
stdout = logging.StreamHandler(sys.stdout)
stdout.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger("default")
logger.setLevel(logging.INFO)
logger.addHandler(stdout)

MAX_WORDS_PER_SENT = 100
MAX_SENT = 15
MAX_VOC_SIZE = 30000
EMBEDDING_DIM = 300
TEST_SPLIT = 1 / 8


#####################################################
# Pre processing                                    #
#####################################################
logger.info("Pre-processsing data.")

dataset = TvTropesMovieSingleDataset()
x1, y1 = dataset.get_dataset("train")
x2, y2 = dataset.get_dataset("val")

reviews = x1 + x2
target = y1 + y2

#####################################################
# Tokenization                                      #
#####################################################
logger.info("Tokenization.")

# Build a Keras Tokenizer that can encode every token
word_tokenizer = Tokenizer(num_words=MAX_VOC_SIZE)
word_tokenizer.fit_on_texts(reviews)

# Construct the input matrix. This should be a nd-array of
# shape (n_samples, MAX_SENT, MAX_WORDS_PER_SENT).
# We zero-pad this matrix (this does not influence
# any predictions due to the attention mechanism.
X = np.zeros((len(reviews), MAX_SENT, MAX_WORDS_PER_SENT), dtype="int32")

for i, review in tqdm(enumerate(reviews)):
    sentences = sent_tokenize(review)
    tokenized_sentences = word_tokenizer.texts_to_sequences(sentences)
    tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=MAX_WORDS_PER_SENT)

    pad_size = MAX_SENT - tokenized_sentences.shape[0]

    if pad_size < 0:
        tokenized_sentences = tokenized_sentences[0:MAX_SENT]
    else:
        tokenized_sentences = np.pad(
            tokenized_sentences,
            ((0, pad_size), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # Store this observation as the i-th observation in
    # the data matrix
    X[i] = tokenized_sentences[None, ...]

# Transform the labels into a format Keras can handle
y = np.array(target)
# We make a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)


#####################################################
# Word Embeddings                                   #
#####################################################
logger.info("Creating embedding matrix using pre-trained vectors.")

# Now, we need to build the embedding matrix. For this we use
# a pretrained (on the wikipedia corpus) 100-dimensional embeddings
# model.

# Load the embeddings from a file
embeddings = {}
with open("data/wiki-news-300d-1M.vec", encoding="utf-8") as file:
    next(file)  # skip header
    for line in tqdm(file):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")

        embeddings[word] = coefs

# Initialize a matrix to hold the word embeddings
embedding_matrix = np.random.random((len(word_tokenizer.word_index) + 1, EMBEDDING_DIM))

# Let the padded indices map to zero-vectors. This will
# prevent the padding from influencing the results
embedding_matrix[0] = 0

# Loop though all the words in the word_index and where possible
# replace the random initalization with the GloVe vector.
for word, index in tqdm(word_tokenizer.word_index.items()):
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


#####################################################
# Model Training                                    #
#####################################################
logger.info("Training the model.")

han_model = HAN(
    MAX_WORDS_PER_SENT,
    MAX_SENT,
    embedding_matrix,
    word_encoding_dim=100,
    sentence_encoding_dim=100,
)

loss = tf.keras.losses.BinaryCrossentropy(name="loss")
# loss = WeightedBinaryCrossEntropy(pos_weight=442475 / 89972, name="loss")
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
han_model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

checkpoint_saver = ModelCheckpoint(
    filepath="./tmp/model.{epoch:02d}-{val_loss:.2f}.hdf5",
    verbose=1,
    save_best_only=True,
)

han_model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_test, y_test),
    # callbacks=[checkpoint_saver],
)
