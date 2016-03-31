import numpy as np
import os
import json
import collections as col
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

TRAINING_PATH = '/media/removable/USB DISK/snli_1.0/snli_1.0_train.jsonl'
DEV_PATH = '/media/removable/USB DISK/snli_1.0/snli_1.0_dev.jsonl'
TEST_PATH = '/media/removable/USB DISK/snli_1.0/snli_1.0_test.jsonl'

TRAINING_VECTORS_PATH = '/home/jack/NLP/csr.npz'
DEV_VECTORS_PATH = '/home/jack/dev_csr.npz'
TEST_VECTORS_PATH = ''

SAVE_TO_FILE = True
LOAD_FROM_FILE = True

STOP = stopwords.words('english')

def save_sparse_vectors(file_name, array, labels):
    np.savez(file_name, data = array.data, indices = array.indices, indptr = array.indptr, shape = array.shape, label_data = labels)

def load_sparse_vectors(file_name):
    loader = np.load(file_name)
    return (csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']), loader['label_data'])

def cross_unigram(sentence1, sentence2, stop_list):
    split1 = [i for i in re.findall("\w+", sentence1.lower()) if i not in stop_list]
    split2 = [i for i in re.findall("\w+", sentence2.lower()) if i not in stop_list]

    cross_unigrams = []
    for word1 in split1:
        for word2 in split2:
            cross_unigrams.append((word1,word2))

    return cross_unigrams

def cross_unigram_counter(sentence1, sentence2):
    unigrams = [hash(i) for i in cross_unigram(sentence1, sentence2, STOP)]
    counter = col.Counter(unigrams)
    return counter

def calculate_vectors(training_path, testing_path):
    pairs = []
    with open(training_path, 'rb') as f:
        for line in f:
            json_pair = json.loads(line)
            pair = {'sentence1':json_pair['sentence1'],
                    'sentence2':json_pair['sentence2'],
                    'label':json_pair['gold_label']}
            pairs.append(pair)

    dev_pairs = []
    with open(testing_path, 'rb') as f:
        for line in f:
            json_pair = json.loads(line)
            pair = {'sentence1':json_pair['sentence1'],
                    'sentence2':json_pair['sentence2'],
                    'label':json_pair['gold_label']}
            dev_pairs.append(pair)


    labels = [pair['label'] for pair in pairs]
    dev_labels = [pair['label'] for pair in dev_pairs]

    counters = [cross_unigram_counter(pair['sentence1'],pair['sentence2']) for pair in pairs]
    dev_counters = [cross_unigram_counter(pair['sentence1'],pair['sentence2']) for pair in dev_pairs]

    vectorizer = DictVectorizer()
    vectorizer.fit(counters)

    vectors = vectorizer.transform(counters)
    dev_vectors = vectorizer.transform(dev_counters)

    if SAVE_TO_FILE:
        save_sparse_vectors('/home/jack/NLP/csr.npz', vectors, labels)
        save_sparse_vectors('/home/jack/NLP/dev_csr.npz', dev_vectors, dev_labels)

    return (vectors, labels, dev_vectors, dev_labels)



if LOAD_FROM_FILE:
    try:
        vectors, labels = load_sparse_vectors(TRAINING_VECTORS_PATH)
        dev_vectors, dev_labels = load_sparse_vectors(DEV_VECTORS_PATH)
    except Exception as e:
        print 'Failed to load from File calculating feature vectors'
        vectors, labels, dev_vectors, dev_labels = calculate_vectors(TRAINING_PATH, DEV_PATH)
else:
    'Computing feature vectors'
    vectors, labels, dev_vectors, dev_labels = calculate_vectors(TRAINING_PATH, DEV_PATH)


#random_state gives the seeed, none seams to always give the same result
perceptron = Perceptron(shuffle=True, n_iter=5, random_state=1000)
perceptron = perceptron.fit(vectors, labels)
predictions = perceptron.predict(dev_vectors)
score = perceptron.score(dev_vectors, dev_labels)
print score


print confusion_matrix(loaded_dev_labels, predictions, labels=['entailment', 'contradiction', 'neutral'])

print classification_report(loaded_dev_labels, predictions,labels=['entailment', 'contradiction', 'neutral'])
