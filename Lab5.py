import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
import nltk
from itertools import product
import time

class Phi():
    def __init__(self):
        self.vectorizer = DictVectorizer()

    def fit(self, training_sentences):
        counts = []
        for sentence in training_sentences:
            words = [pair[0] for pair in sentence]
            tags = [pair[1] for pair in sentence]
            count = self.extract_freatures(words, tags)
            counts.append(count)

        #Fit the dictvectorizer to the data
        #Fit expects a list of dictonaries but we just give it the one we constructed
        self.vectorizer.fit(counts)

    def transform(self, word_sequence, tag_sequence):
        #Extract the feature count
        count = self.extract_freatures(word_sequence, tag_sequence)

        #Convert the count to a sparse vector using dictVectorizer
        vector = self.vectorizer.transform(count)

        return vector

    def extract_freatures(self, word_sequence, tag_sequence):
        #Force them to be the same length
        length = len(word_sequence)

        #Append end to tags
        tag_sequence.append('END')

        #Create a list of word-tag and tag-tag features
        features = []
        for i in range(0,length):
            features.append((word_sequence[i],tag_sequence[i]))
            features.append((tag_sequence[i],tag_sequence[i+1]))

        #Create a count out of the list of features
        count = Counter(features)

        return count

    def inverse_transform(self, vector):
        #Convert the vector to a count using dictVectorizer
        counts = self.vectorizer.inverse_transform(vector)

        #inverse_transform returns a list of counts but we only gave it one vector
        return counts[0]

def structured_perceptron_predict(word_sequence, weights, phi, unique_tags):
    #length = len(word_sequence)

    #Converts the iterable into a list in order to return the best one
    #Might cause overflow error for long word sequences >10
    #tag_sequences = list(product(unique_tags, repeat=length))

    #scores = []
    #for tag_sequence in tag_sequences:
    #   tag_sequence = list(tag_sequence)
    #   vector = phi.transform(word_sequence, tag_sequence).toarray()
    #   score = np.dot(weights.toarray(), vector.T)
    #   scores.append(score)

    #return list(tag_sequences[np.argmax(scores)])
    #print word_sequence
    return viterbi(word_sequence, weights, phi, unique_tags)

def structured_perceptron_train(examples, weights, phi, unique_tags, iterations=1, average_weights=False, shuffle=True):
    for i in range(0,iterations):
        if shuffle:
            np.random.shuffle(examples)

        #if AVERAGE_WEIGHTS:
        #    weights_avg = csr_matrix((1,examples[0].shape[1]))

        # count = 0
        for example in examples:
            # print count
            word_sequence = example[0]
            tag_sequence = example[1]

            predicted_tag_sequence = structured_perceptron_predict(word_sequence, weights, phi, unique_tags)

            if predicted_tag_sequence != tag_sequence:
                weights += phi.transform(word_sequence, tag_sequence) - phi.transform(word_sequence, predicted_tag_sequence)

            # count += 1

            #if bool(prediction) != label:
            #    if prediction == 1:
            #        weights[0] += vector
            #        weights[1] -= vector
            #    elif prediction == 0:
            #        weights[1] += vector
            #        weights[0] -= vector

            #    if AVERAGE_WEIGHTS:
            #        count += 1
            #        # Rolling average using
            #        # avg -= avg / count
            #        # avg += new_sample / count
            #        weights_avg -= weights_avg / count
            #        weights_avg += weights / count

    #if AVERAGE_WEIGHTS:
    #    #average = weights_sum / float(iterations * len(examples))
    #    return weights_avg
    #else:
    return weights

def structured_perceptron_test(word_sequences, weights, phi, unique_tags):
    predictions = []
    for word_sequence in word_sequences:

        prediction = structured_perceptron_predict(word_sequence, weights, phi, unique_tags)
        predictions.append(prediction)

    return predictions

def viterbi(word_sequence, weights, phi, unique_tags):
    trellis = [{}]

    #Set the start scores in the trellis
    for tag in unique_tags:
        trellis[0][tag] = (0, -1)

    #Iterate through the trellis updating the scores and back pointers as you go
    for t in range(1,len(word_sequence)):
        trellis.append({})
        for current_tag in unique_tags:
            scores = []
            for previous_tag in unique_tags:
                vector = phi.transform([word_sequence[t-1], word_sequence[t]], [previous_tag, current_tag])
                score = np.dot(weights.toarray(), vector.toarray().T) + trellis[t-1][previous_tag][0]
                scores.append(score[0][0])

            max_score = max(scores)
            back_pointer = np.argmax(scores)
            trellis[t][current_tag] = (max_score, back_pointer)

    #Calculate the tag index of the higest scoring tag in the final row of the trellis
    index = len(trellis)-1
    predicted_tags = []
    items = trellis[index].items()
    scores = [pair[1][0] for pair in items]
    next_tag_index = np.argmax(scores)

    #Follow the back pointers until you reach the end of the trellis or reach the start tag of -1
    while index >= 0 and next_tag_index >= 0:
        items = trellis[index].items()
        current_item = items[next_tag_index]
        predicted_tags.append(current_item[0])
        next_tag_index = current_item[1][1]
        index -= 1

    predicted_tags.reverse()
    return predicted_tags



corpus = nltk.corpus.brown.tagged_sents(categories='news',tagset='universal')
training = corpus[0:3000]
testing = corpus[3001:]


unique_tags = set()
for sentence in training:
    tags = [pair[1] for pair in sentence]
    unique_tags = unique_tags | set(tags)


phi = Phi()
phi.fit(training)

MAX_LENGTH = 7
print 'Max Sentence Length: ', MAX_LENGTH

reduced_training = []
for sentence in training:
    words = [pair[0] for pair in sentence]
    if len(words) <= MAX_LENGTH:
        reduced_training.append(sentence)

print 'Training Length: ', len(reduced_training)

training_examples = []
for sentence in reduced_training:
    words = [pair[0] for pair in sentence]
    tags = [pair[1] for pair in sentence]
    training_examples.append((words, tags))


reduced_testing = []
for sentence in testing:
    words = [pair[0] for pair in sentence]
    if len(words) <= MAX_LENGTH:
        reduced_testing.append(sentence)

print 'Testing Length: ', len(reduced_testing)

testing_examples = []
for sentence in reduced_training:
    words = [pair[0] for pair in sentence]
    tags = [pair[1] for pair in sentence]
    testing_examples.append((words, tags))



print 'Training Perceptron'
weights = csr_matrix((1,11354))
start = time.time()
new_weights = structured_perceptron_train(training_examples, weights, phi, unique_tags, 1, False, False)
end = time.time()

time_taken = end - start
print 'Time taken: ', time_taken


print 'Testing Perceptron'
words = [example[0] for example in testing_examples]
start = time.time()
predictions = structured_perceptron_test(words, new_weights, phi, unique_tags)
end = time.time()

time_taken = end - start
print 'Time taken: ', time_taken


print 'Scoring Perceptron'
correct_word = 0
total_word = 0
correct_sentence = 0
for i in range(0,len(predictions)):
    real = testing_examples[i][1]
    predicted = predictions[i]

    if real == predicted:
        correct_sentence += 1

    for j in range(0,len(real)):
        total_word += 1
        if real[j] == predicted[j]:
            correct_word += 1

print 'Sentence Accuracy: ', correct_sentence / float(len(predictions))
print 'Word Accuracy: ', correct_word / float(total_word)
