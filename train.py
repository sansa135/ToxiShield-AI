import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load Dataset
df = pd.read_csv("data/train.csv")

# Input and Output
X = df["comment_text"].astype(str)

y = df["toxic"]

# Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(X_seq, maxlen=200)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = Sequential()

model.add(Embedding(5000, 128, input_length=200))

model.add(LSTM(64))

model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=2,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save
model.save("model/toxic_model.h5")

print("Model Trained Successfully")

