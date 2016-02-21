import re
import collections as col
import numpy as np
from random import shuffle
import os

FOLDER_PATH = '/software/Python/NLP/txt_sentoken/'
SAVE_FILE_NAME = 'review_vectors'
LOAD_FROM_FILE = True
SAVE_TO_FILE = True
SHUFFLE = True

def create_bag_of_words(file_path):
    file = open(FOLDER_PATH + file_path)

    raw = file.read().lower()
    split = re.findall("\w+", raw)

    bag = col.Counter(split)

    file.close()
    return bag

def convert_to_standard_vector(key_set, bag):
    vector = []
    for key in key_set:
        if key in bag[0]:
            vector.append(bag[0][key])
        else:
            vector.append(0)

    return (np.array(vector), bag[1])

def compute_bags_of_words(path_to_dir, key_set, positive_review):
    bags = []

    for path in os.listdir(FOLDER_PATH + path_to_dir):
        complete_path = path_to_dir + path
        bag = create_bag_of_words(complete_path)
        bags.append((bag, positive_review))
        key_set.update(bag.keys())

    return bags

def compute_review_vectors():
    key_set = set()
    bags = []
    vectors = []

    bags += compute_bags_of_words('neg/', key_set, False)
    bags += compute_bags_of_words('pos/', key_set, True)

    for bag in bags:
        vector = convert_to_standard_vector(key_set, bag)
        vectors.append(vector)

    print 'Finished computing vectors from txt files.'

    if SAVE_TO_FILE:
        np.save(FOLDER_PATH + SAVE_FILE_NAME, vectors)
        print 'Saved vectors to ' + SAVE_FILE_NAME + '.npy'

    return vectors

def perceptron_train(examples, weights, iterations=1):
    for i in range(0,iterations):
        for example in examples:
            vector = example[0]
            label = example[1]

            raw_prediction = np.dot(weights, vector)
            prediction = raw_prediction > 0

            if raw_prediction == 0:
                weights = weights + vector
            elif label != prediction:
                if label:
                    weights = weights + vector
                else:
                    weights = weights - vector

    return weights

def perceptron_test(examples, weights):
    predictions = []
    for example in examples:
        vector = example[0]
        raw_prediction = np.dot(weights, vector)
        prediction = raw_prediction > 0
        predictions.append(prediction)

    return predictions

#train = vectors[0:800] + vectors[1000:1800]
#test = vectors[800:1000] + vectors[1800:2000]
# print type(vectors)
# print type(train)
# print type(test)
#
#
# train_pos = 0
# train_neg = 0
#
# for i in range(0,len(train)):
#     if train[i][1]:
#         train_pos += 1
#     else:
#         train_neg += 1
#
# print 'Train total: ' + str(len(train))
# print 'Train Positive: ' + str(train_pos)
# print 'Train Negative: ' + str(train_neg)
#
# test_pos = 0
# test_neg = 0
#
# for i in range(0,len(test)):
#     if test[i][1]:
#         test_pos += 1
#     else:
#         test_neg += 1
#
# print 'Test total: ' + str(len(test))
# print 'Test Positive: ' + str(test_pos)
# print 'Test Negative: ' + str(test_neg)

if LOAD_FROM_FILE:
    try:
        print 'Loading vectors from ' + SAVE_FILE_NAME + '.npy'
        vectors = np.load(FOLDER_PATH + SAVE_FILE_NAME + '.npy')
    except IOError as e:
        print 'Failed to load vectors from ' + SAVE_FILE_NAME + '.npy'
        print 'Computing vectors from txt files.'
        vectors = compute_review_vectors()
else:
    vectors = compute_review_vectors()

train = np.concatenate((vectors[0:800], vectors[1000:1800]))
test = np.concatenate((vectors[800:1000], vectors[1800:2000]))

if SHUFFLE:
    np.random.shuffle(train)
    np.random.shuffle(test)

weights = np.zeros(len(vectors[0][0]))
weights = perceptron_train(train, weights, 20)
predictions = perceptron_test(test, weights)

correct = 0
pos = 0
neg = 0

for i in range(0,len(predictions)):
    prediction = predictions[i]
    actual = test[i][1]

    if prediction:
        pos += 1

    if not prediction:
        neg += 1

    if prediction and actual:
        correct += 1
    elif not prediction and not actual:
        correct += 1

print 'Correct: ' + str(correct)
print 'Positive: ' + str(pos)
print 'Negative: ' + str(neg)

accuracy = correct / float(len(predictions))

print 'Accuracy: ' + str(accuracy)
