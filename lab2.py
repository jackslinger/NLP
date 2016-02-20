
# coding: utf-8

# In[8]:

import re
import collections as col
import numpy as np
from random import shuffle
import os

#np.set_printoptions(threshold='nan')


# In[9]:

FOLDER_PATH = '/home/jack/Downloads/txt_sentoken/'


# In[10]:

def create_bag_of_words(file_path):
    file = open(FOLDER_PATH + file_path)
    
    raw = file.read().lower()
    split = re.findall("\w+", raw)
    
    bag = col.Counter(split)
    
    file.close()
    return bag
    


# In[11]:

def convert_to_standard_vector(key_set, bag):
    vector = []
    for key in key_set:
        if key in bag[0]:
            vector.append(bag[0][key])
        else:
            vector.append(0)
    
    return (np.array(vector), bag[1])


# In[12]:

def compute_bags_of_words(path_to_dir, key_set, positive_review):
    bags = []
    
    for path in os.listdir(FOLDER_PATH + path_to_dir):
        complete_path = path_to_dir + path
        bag = create_bag_of_words(complete_path)
        bags.append((bag, positive_review))
        key_set.update(bag.keys())
    
    return bags


# In[13]:

key_set = set()
bags = []
vectors = []

bags += compute_bags_of_words('neg/', key_set, False)
bags += compute_bags_of_words('pos/', key_set, True)

for bag in bags:
    vector = convert_to_standard_vector(key_set, bag)
    vectors.append(vector)


# In[54]:

train = vectors[0:800] + vectors[1000:1800]
test = vectors[800:1000] + vectors[1800:2000]


# In[75]:

np.save(FOLDER_PATH + 'stored_data', vectors)


# In[76]:

read_array = np.load(FOLDER_PATH + 'stored_data.npy')


# In[80]:

print read_array[1020][1]


# In[55]:

def perceptron_train(examples, weights):
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


# In[56]:

def perceptron_test(examples, weights):
    predictions = []
    for example in examples:
        vector = example[0]
        raw_prediction = np.dot(weights, vector)
        prediction = raw_prediction > 0
        predictions.append(prediction)
    
    return predictions


# In[65]:

shuffle(train)
shuffle(test)

weights = np.zeros(len(key_set))

weights = perceptron_train(train, weights)

predictions = perceptron_test(test, weights)


# In[66]:

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

print correct
print pos
print neg

accuracy = correct / float(len(predictions))

print accuracy
    


# In[270]:

assert len(predictions) == len(test)

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for i in range(0,len(predictions)):
    prediction = predictions[i]
    actual = test[i][1]
    
    if prediction and actual:
        true_positive += 1
    elif prediction and not actual:
        false_positive += 1
    elif not prediction and actual:
        false_negative += 1
    elif not prediction and not actual:
        true_negative += 1

print true_positive
print true_negative
print false_positive
print false_negative
        
accuracy = (true_positive + true_negative) / float(len(predictions))
        
pos_precision = true_positive / float(true_positive + false_positive)
pos_recall = true_positive / float(true_positive + false_negative)
pos_fscore = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)

neg_precision = true_negative / float(true_negative + false_negative)
neg_recall = true_negative / float(true_negative + false_positive)
neg_fscore = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)

print "Accuracy: " + str(accuracy)
print

print "Positive Reviews"
print "Precision: " + str(pos_precision)
print "Recall: " + str(pos_recall)
print "F-Score: " + str(pos_fscore)
print

print "Negative Reviews"
print "Precision: " + str(neg_precision)
print "Recall: " + str(neg_recall)
print "F-Score: " + str(neg_fscore)


# In[ ]:



