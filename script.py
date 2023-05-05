import sys
language = sys.argv[1]    ## getting language from commandline  

from library.preprocess import *
from library.models import *



## ---------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------- ##
## Folder structure: data/{language}/

## Input Files: tokenized_train_text.pkl, tokenized_val_text.pkl, tokenized_test_text.pkl
## Output Labels: train_labels.pkl, val_labels.pkl, test_labels.pkl

## Data transform output: padded_train_data.pkl, padded_val_data.pkl, padded_test_data.pkl
##                        ohe_train_labels.pkl, ohe_val_labels.pkl, ohe_test_labels.pkl

## RNN output: RNN model file, RNN_prediction.pkl, RNN_test_evaluation.txt

## CNN output: CNN model file, CNN_prediction.pkl, CNN_test_evaluation.txt
## ---------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------- ##



## ---------- Pre-processing data  ---------- 
path = 'data/'
max_len = 50

transform = DataTransformation(path, language, max_len)
max_words, vocab_len = transform.run()


## ---------- CNN model  ----------
path = 'data/'
top_words = vocab_len      
epochs=25
batch_size=100

cnn = CNN_model(path, language, top_words, max_words, epochs, batch_size)
cnn.run()


## ---------- RNN model  ----------
path = 'data/'    
epochs=25
batch_size=100

rnn = RNN_model(path, language, epochs, batch_size)
rnn.run()