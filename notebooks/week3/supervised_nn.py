from keras.datasets import imdb
import numpy as np


def describe_review(m, features, labels):
    print('Item number: ', m, '\n')
    # print label
    print("Label:", labels[m], '\n')
    # print encoded features
    print("Encoded: ", features[m], '\n')
    # Get index of words
    index = imdb.get_word_index()
    # Reverse key and index
    reverse_index = dict([(value, key) for (key, value) in index.items()]) 
    # decoded example 
    decoded = " ".join( [reverse_index.get(i - 3, "#") for i in features[m]] )
    print("Decoded: ", decoded )
    
    
def load_imdb(top_words, n):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
    features = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)
    
    print ('Dataset Loaded!')
    return features[0:n], labels[0:n]


def split_dataset(features, labels, r):
    # r is the proportion of features for the test set
    l = int(len(features)*r)
    test_x = features[:l]
    test_y = labels[:l]
    train_x = features[l:]
    train_y = labels[l:]
    
    return train_x, train_y, test_x, test_y