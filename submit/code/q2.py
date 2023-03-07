from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the FIGER dataset
X, y = load_figer_data()

# Convert the labels to one-hot encoding
num_labels = len(set(y))
y = to_categorical(y, num_classes=num_labels)

# Pad the input sequences to a fixed length
maxlen = 100
X = pad_sequences(X, maxlen=maxlen)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen))
model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the LSTM model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
precision = metrics.precision_score(y_test.argmax(-1), y_pred, average='weighted')
recall = metrics.recall_score(y_test.argmax(-1), y_pred, average='weighted')
f1_score = metrics.f1_score(y_test.argmax(-1), y_pred, average='weighted')
