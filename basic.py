#import imdb
from keras.datasets import imdb

#load data, reviews contain only top 10,000 most common words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index() #get words->index dictionary
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #reverse dictionary to retrieve words by index

#if you'd like to see what the reviews actually are, uncomment the following line to see the first review decoded
#decoded_review = " ".join([reverse_word_index.get(index-3, '?') for index in train_data[0]])

#import numpy
import numpy as np

#method to hot-encode a 2-dim tensor
#data cannot be fed to the model in String representation.
#converting to integers based on index in the standard english dictionary is also not feasible.
#one-hot encoding is the best suited representation for this problem
def hot_encode_matrix(sequences, vector_dimensions=10000):
    results = np.zeros((len(sequences), vector_dimensions))
    for index, sequence in enumerate(sequences):
        results[index, sequence] = 1
    return results

#hot encode training data
x_train = hot_encode_matrix(train_data)
x_test = hot_encode_matrix(train_labels)

#hot encode testing data
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#set validation set of 10000 elements apart
x_val = x_train[:10000]
y_val = y_train[:10000]

#get training data separated from validation data
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

#import models and layers modules
from keras import models, layers

#create a Sequential model
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,))) #first layer with 16 hidden units
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid")) #output layer

#compile model
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#fit the model!
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=4, #running the model beyond 4 epochs causes overfitting and raises validation loss
    batch_size=512,
    validation_data=(x_val, y_val)
)
