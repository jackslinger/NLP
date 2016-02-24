import re
import collections as col
import numpy as np
from random import shuffle
import os
import pickle
import time

#FOLDER_PATH = '/software/Python/NLP/txt_sentoken/'
FOLDER_PATH = os.getcwd() + '/'
#FOLDER_PATH = '/home/jack/Downloads/txt_sentoken/'
SAVE_FILE_NAME = 'review_vectors'
KEYSET_FILE_NAME = 'keyset_vector'
LOAD_FROM_FILE = False
SAVE_TO_FILE = False
SHUFFLE = True
ITERATIONS = 8
AVERAGE_WEIGHTS = True

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
    print 'Starting to compute vectors for txt files'

    #Use a set to ensure that no words are duplicated in the key set
    key_set = set()
    bags = []
    vectors = []

    bags += compute_bags_of_words('neg/', key_set, False)
    bags += compute_bags_of_words('pos/', key_set, True)

    #Convert key_set into a list to enforce a set order
    key_set = list(key_set)

    for bag in bags:
        vector = convert_to_standard_vector(key_set, bag)
        vectors.append(vector)

    print 'Finished computing vectors from txt files.'

    if SAVE_TO_FILE:
        np.save(FOLDER_PATH + SAVE_FILE_NAME, vectors)
        with open(FOLDER_PATH + KEYSET_FILE_NAME, 'w') as f:
            pickle.dump(key_set,f)
        print 'Saved vectors to ' + SAVE_FILE_NAME + '.npy'

    return vectors, key_set

def perceptron_predict(vector, weights):
    scores = []
    for class_weight in weights:
        scores.append(np.dot(class_weight, vector))

    return np.argmax(scores)

def perceptron_train(examples, weights, iterations=1, average_weights=False):
    for i in range(0,iterations):
        if SHUFFLE:
            np.random.shuffle(examples)

        if AVERAGE_WEIGHTS:
            weights_avg = np.zeros((2,len(examples[0][0])))

        count = 0
        for example in examples:
            vector = example[0]
            label = example[1]

            prediction = perceptron_predict(vector, weights)

            if bool(prediction) != label:
                if prediction == 1:
                    weights[0] += vector
                    weights[1] -= vector
                elif prediction == 0:
                    weights[1] += vector
                    weights[0] -= vector

                if AVERAGE_WEIGHTS:
                    count += 1
                    # Rolling average using
                    # avg -= avg / count
                    # avg += new_sample / count
                    weights_avg -= weights_avg / count
                    weights_avg += weights / count

    if AVERAGE_WEIGHTS:
        #average = weights_sum / float(iterations * len(examples))
        return weights_avg
    else:
        return weights

def perceptron_test(examples, weights):
    predictions = []
    for example in examples:
        vector = example[0]

        prediction = perceptron_predict(vector, weights)
        predictions.append(bool(prediction))

    return predictions

def compute_accuracy(predictions, examples, print_to_screen=False):
    correct = 0
    pos = 0
    neg = 0

    for i in range(0,len(predictions)):
        prediction = predictions[i]
        actual = examples[i][1]

        if prediction:
            pos += 1

        if not prediction:
            neg += 1

        if prediction and actual:
            correct += 1
        elif not prediction and not actual:
            correct += 1

    accuracy = correct / float(len(predictions))

    if print_to_screen:
        print 'Correct: ' + str(correct)
        print 'Positive: ' + str(pos)
        print 'Negative: ' + str(neg)
        print 'Accuracy: ' + str(accuracy)

    return accuracy

def print_top_words(weights, key_set, limit):
    sorted_neg_weights = np.argsort(weights[0])
    length = len(sorted_neg_weights)
    print 'Top 10 Negative Words'
    for i in sorted_neg_weights[length-10:length]:
        print key_set[i]

    print '-' * 20

    sorted_pos_weights = np.argsort(weights[1])
    length = len(sorted_pos_weights)
    print 'Top 10 Positive Words'
    for i in sorted_pos_weights[length-10:length]:
        print key_set[i]




print '-' * 20
print 'Perceptron settings'
print 'Load from file: ' + str(LOAD_FROM_FILE)
print 'Save to file: ' + str(SAVE_TO_FILE)
print 'Shuffling: ' + str(SHUFFLE)
print 'Averaging weights: ' + str(AVERAGE_WEIGHTS)
print 'Iterations: ' + str(ITERATIONS)
print '-' * 20

if LOAD_FROM_FILE:
    try:
        print 'Loading vectors from ' + SAVE_FILE_NAME + '.npy'
        vectors = np.load(FOLDER_PATH + SAVE_FILE_NAME + '.npy')
        print 'Loading key set from ' + KEYSET_FILE_NAME
        key_set = pickle.load(open(FOLDER_PATH + KEYSET_FILE_NAME, 'rb'))
    except IOError as e:
        print 'Failed to load from file.'
        #print 'Failed to load vectors from ' + SAVE_FILE_NAME + '.npy'
        print 'Computing vectors from txt files.'
        vectors, key_set = compute_review_vectors()
else:
    vectors, key_set = compute_review_vectors()

train = np.concatenate((vectors[0:800], vectors[1000:1800]))
test = np.concatenate((vectors[800:1000], vectors[1800:2000]))

weights = np.zeros((2,len(vectors[0][0])))

print 'Training the perceptron'
weights = perceptron_train(train, weights, ITERATIONS)
predictions = perceptron_test(test, weights)

print '-' * 20
compute_accuracy(predictions, test, True)

print '-' * 20
print_top_words(weights, key_set, 10)

# times = []
# for iterations in range(10,11):
#     start = time.time()
#
#     weights = np.zeros((2,len(vectors[0][0])))
#     weights = perceptron_train(train, weights, iterations)
#     predictions = perceptron_test(test, weights)
#
#     compute_accuracy(predictions, test)
#
#     end = time.time()
#
#     time_taken = end - start
#     print time_taken
#     times.append(time_taken)
#
# print times


# iter_results = []
# for iterations in range(0,1):
#     results = []
#     for run in range(0,20):
#         weights = np.zeros((2,len(vectors[0][0])))
#         weights = perceptron_train(train, weights, iterations)
#         predictions = perceptron_test(test, weights)
#
#         results.append(compute_accuracy(predictions, test, True))
#
#         #print_top_words(weights, key_set, 10)
#
#     print results
#     print np.mean(results)
#     iter_results.append(results)
