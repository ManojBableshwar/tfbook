import argparse
import os
import numpy as np

import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser("train")
parser.add_argument("--data", type=str, help="Path to training data")
parser.add_argument("--model", type=str, help="Path to saved model")
parser.add_argument("--score", type=str, help="Path to scoring data")
parser.add_argument("--predict_count", type=str, help="number of words to predict")
parser.add_argument("--output", type=str, help="output data file")
args = parser.parse_args()

model = tf.keras.models.load_model(args.model)

tokenizer = Tokenizer()

data = open(os.path.join(args.data, os.listdir(args.data)[0])).read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])

score_data = open(os.path.join(args.score, os.listdir(args.score)[0])).read().lower().split("\n")
next_words = int(args.predict_count)
score_output = []

print (score_data)

for seed_text in score_data:
	print(seed_text)
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		#predicted = model.predict_classes(token_list, verbose=0)
		predicted = np.argmax(model.predict(token_list), axis=-1)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	print(seed_text)
	score_output.append(seed_text)

with open(os.path.join(args.output,'irish-output.txt'), 'w') as temp_file:
    for item in score_output:
        temp_file.write("%s\n" % item)