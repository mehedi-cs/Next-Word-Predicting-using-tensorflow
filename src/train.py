import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

with open('data/input_text.txt', 'r', encoding='utf-8') as f:
    data = f.read()
  
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

input_sequences = []
for sentence in data.split('.'):
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

max_length = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')
X = padded_sequences[:, :-1]
Y = padded_sequences[:, -1]
Y = to_categorical(Y, num_classes=len(tokenizer.word_index)+1)

vocab_size = len(tokenizer.word_index) + 1
model = Sequential([
    Embedding(vocab_size, 100, input_length=max_length-1),
    LSTM(150),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, Y, epochs=100)

model.save('models/next_word_model.h5')
