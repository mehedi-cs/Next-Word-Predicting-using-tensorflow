import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

model = tf.keras.models.load_model('models/next_word_model.h5')

from tensorflow.keras.preprocessing.text import Tokenizer
with open('data/input_text.txt', 'r', encoding='utf-8') as f:
    data = f.read()
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

max_length = max([len(tokenizer.texts_to_sequences([s])[0]) for s in data.split('.')])

def predict_next_words(seed_text, n_words=2):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += ' ' + word
                break
    return seed_text

print(predict_next_words("ai", 2))
