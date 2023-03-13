import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from figer_classes import *

# Load data
with open('../../data/dev.json') as f:
    data = json.load(f)

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define feature functions
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        prev_word = sent[i-1]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isupper()': prev_word.isupper()
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        next_word = sent[i+1]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isupper()': next_word.isupper()
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return sent['tags']

# def sent2multilabels(sent):
#     label_array = np.zeros((len(sent['sent']), len(FIGER_TAGS)), dtype=np.int8)
#     for i, tag_list in enumerate(sent['tags']):
#         if tag_list == 'O':
#             continue
#         for tag in tag_list:
#             # get index of tag in FIGER class list
#             idx = FIGER_TAGS.index(tag)
#             label_array[i, idx] = 1
#     return [str(i) for i in label_array.tolist()]

# Convert data to features and labels
X_train = [sent2features(sent['sent']) for sent in train_data]
y_train = [sent2labels(sent) for sent in train_data]
# X_test = [sent2features(sent['sent']) for sent in test_data]
# y_test = [sent2labels(sent) for sent in test_data]

# print(X_train[0], y_train[0])

# Train CRF model
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

# # Make predictions on test data
# y_pred = crf.predict(X_test)

# # Print classification report
# labels = list(set(tag for sent in y_test for tag in sent))
# print(flat_classification_report(y_test, y_pred, labels=labels))
