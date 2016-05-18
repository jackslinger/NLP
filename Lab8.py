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
import matplotlib.pyplot as plt

# Change these paths to work on your system!
TRAINING_PATH = 'D:/software/Python/snli_1.0/snli_1.0_train.jsonl'
DEV_PATH = 'D:/software/Python/snli_1.0/snli_1.0_dev.jsonl'
TEST_PATH = 'D:/software/Python/snli_1.0/snli_1.0_test.jsonl'

STOP = stopwords.words('english')

def pos_stoplist(pos):
    return pos == 'NN' or pos == 'JJ' or 'VB' in pos

def pos_strip(sentence_parse, stoplist_function):
    m = re.finditer('\((\w+) (\w+)\)', sentence_parse)
    tags = [(match.group(2), match.group(1)) for match in m]
    return tags

def word_strip(sentence):
    return [i for i in re.findall("\w+", sentence.lower())]

def extract_pair(line):
    json_pair = json.loads(line)
    pair = {'sentence1': word_strip(json_pair['sentence1']),
            'sentence1_pos': pos_strip(json_pair['sentence1_parse'], pos_stoplist),
            'sentence2': word_strip(json_pair['sentence2']),
            'sentence2_pos': pos_strip(json_pair['sentence2_parse'], pos_stoplist),
            'label': json_pair['gold_label']}
    return pair

# The following methods are the different ways that can be used to train a model
def cross_unigram_counter(pair):
    cross_unigrams = []
    for word1 in pair['sentence1']:
        for word2 in pair['sentence2']:
            cross_unigrams.append((word1, word2))

    return col.Counter(cross_unigrams)

def cross_unigram_stoplist_counter(pair):
    sentence1 = (word for word in pair['sentence1'] if word not in STOP)
    sentence2 = (word for word in pair['sentence2'] if word not in STOP)

    cross_unigrams = []
    for word1 in sentence1:
        for word2 in sentence2:
            cross_unigrams.append((word1, word2))

    return col.Counter(cross_unigrams)

def cross_unigram_pos_stoplist_counter(pair):
    sentence1 = (match for match in pair['sentence1_pos'] if pos_stoplist(match[1]))
    sentence2 = (match for match in pair['sentence2_pos'] if pos_stoplist(match[1]))

    cross_unigrams = []
    for match1 in sentence1:
        for match2 in sentence2:
            cross_unigrams.append((match1, match2))

    return col.Counter(cross_unigrams)

def cross_unigram_pos_counter(pair):
    cross_unigrams = []
    for word1 in pair['sentence1_pos']:
        for word2 in pair['sentence2_pos']:
            cross_unigrams.append((word1, word2))

    return col.Counter(cross_unigrams)

def cross_unigram_pos_justwords_counter(pair):
    cross_unigrams = []
    for word1 in pair['sentence1_pos']:
        for word2 in pair['sentence2_pos']:
            cross_unigrams.append((word1[0], word2[0]))

    return col.Counter(cross_unigrams)

def extract_pairs(file_path):
    with open(file_path, 'rb') as f:
        for line in f:
            yield extract_pair(line)

def extract_counters(pairs, feature_extractor):
    for pair in pairs:
        yield feature_extractor(pair)

def calculate_vectors(training_path, dev_path, testing_path, feature_extractor):
    print 'Calculating Labels'
    labels = [pair['label'] for pair in extract_pairs(training_path)]
    dev_labels = [pair['label'] for pair in extract_pairs(dev_path)]
    test_labels = [pair['label'] for pair in extract_pairs(testing_path)]

    vectorizer = DictVectorizer()

    print 'Training Vectorizer'
    vectors = vectorizer.fit_transform(extract_counters(extract_pairs(training_path), feature_extractor))
    print 'Vectorizing unigrams'
    dev_vectors = vectorizer.transform(extract_counters(extract_pairs(dev_path), feature_extractor))
    test_vectors = vectorizer.transform(extract_counters(extract_pairs(testing_path), feature_extractor))

    return (vectors, labels, dev_vectors, dev_labels, test_vectors, test_labels)

# Change cross_unigram_pos_counter to the method you whish to train
print 'Training Model for POS Uni-grams'
print 'Please wait this may take a few minutes to run.'
vectors, labels, dev_vectors, dev_labels, test_vectors, test_labels = calculate_vectors(TRAINING_PATH, DEV_PATH, TEST_PATH, cross_unigram_pos_counter)
print 'Vector Shape: ', vectors.shape

perceptron = Perceptron(shuffle=True, n_iter=5, random_state=1000)
print 'Training Perceptron'
perceptron = perceptron.fit(vectors, labels)
print 'Testing perceptron'
predictions = perceptron.predict(test_vectors)
print 'Train: ', perceptron.score(vectors, labels)
print 'Dev: ', perceptron.score(dev_vectors, dev_labels)
print 'Test: ', perceptron.score(test_vectors, test_labels)

print classification_report(test_labels, predictions, labels=['neutral', 'entailment', 'contradiction'])
