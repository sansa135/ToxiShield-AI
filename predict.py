from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model("model/toxic_model.h5")

tokenizer = Tokenizer(num_words=5000)

text = ["I hate you"]

seq = tokenizer.texts_to_sequences(text)

pad = pad_sequences(seq, maxlen=200)

prediction = model.predict(pad)

print(prediction)
