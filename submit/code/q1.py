import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

# Load the FIGER dataset
X, y = load_figer_data()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CRF model with the features and loss function
crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                           c1=0.1,
                           c2=0.1,
                           max_iterations=100,
                           all_possible_transitions=True)
crf.fit(X_train, y_train)

# Predict the labels for the testing set
y_pred = crf.predict(X_test)

# Compute the evaluation metrics
precision = metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=labels)
recall = metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=labels)
f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
