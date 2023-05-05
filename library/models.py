import pickle
import numpy as np

from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Embedding
from keras.layers import LSTM, SpatialDropout1D, Bidirectional
from keras.preprocessing import sequence



class CNN_model():
    
    def __init__(self, path, language, top_words, max_words, epoch, batch_size, level, mode):
        super().__init__()
        self.path = path+language+'/'+level+'/'+mode+'/'
        self.top_words = top_words
        self.max_words = max_words
        self.epochs = epoch
        self.batch_size = batch_size

        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def write_txt(file_name, file_object, path):
            with open(path+file_name, 'w') as f:
                f.write(file_object)
        
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)
                
        def extract_data(data, _type='text'):
            text = list(map(itemgetter(1), data))
            if _type == 'text':
                ids = list(map(itemgetter(0), data))
                return ids, text
            return text
                
        def build_model(top_words, max_words):
            model = Sequential()      # initilaizing the Sequential nature for CNN model
            # Adding the embedding layer which will take  input and provide a 32 dimensional output
            model.add(Embedding(top_words, 32, input_length=max_words))
            model.add(Conv1D(32, 3, padding='same', activation='relu'))
            model.add(MaxPooling1D())
            model.add(Flatten())
            model.add(Dense(250, activation='relu'))
            model.add(Dense(2, activation='sigmoid'))
            return model
        
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## ------- reading raw data ------- 
        data = read_pickle('padded_train_data.pkl', self.path)
        train_ids, X_train = extract_data(data)
        data = read_pickle('padded_val_data.pkl', self.path)
        val_ids, X_val = extract_data(data)
        data = read_pickle('padded_test_data.pkl', self.path)
        test_ids, X_test = extract_data(data)

        data = read_pickle(f'train_{mode}_labels.pkl', self.path)
        Y_train = extract_data(data, 'labels')
        data = read_pickle(f'val_{mode}_labels.pkl', self.path)
        Y_val = extract_data(data, 'labels')
        
        
        ## ------- building CNN model -------
        model = build_model(self.top_words, self.max_words)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=self.epochs, batch_size=self.batch_size)
        
        pred = model.predict(X_test)
        
        ## ------- writing files -------
        model.save(self.path+f'CNN_{level}-level_{mode}-label')
        write_pickle(f'CNN_{level}-level_{mode}-label_prediction.pkl', list(zip(test_ids,pred)), self.path)
          
        
        
class RNN_model():
    
    def __init__(self, path, language, epochs, batch_size):
        super().__init__()
        self.path = path+language+'/'
        self.epochs = epochs
        self.batch_size = batch_size

        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def write_txt(file_name, file_object, path):
            with open(path+file_name, 'w') as f:
                f.write(file_object)
        
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)
                
        def build_model(input_dim):
            model = Sequential()
            model.add(Embedding(232337, 100, input_length=input_dim))
            model.add(SpatialDropout1D(0.2))
            model.add(Bidirectional(LSTM(20, dropout=0.2, recurrent_dropout=0.2)))
            model.add(Dense(2, activation='softmax'))
            return model
        
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## ------- reading raw data ------- 
        X_train = read_pickle('padded_train_data.pkl', self.path)
        X_val = read_pickle('padded_val_data.pkl', self.path)
        X_test = read_pickle('padded_test_data.pkl', self.path)
        Y_train = read_pickle('ohe_train_labels.pkl', self.path)
        Y_val = read_pickle('ohe_val_labels.pkl', self.path)
        Y_test = read_pickle('ohe_test_labels.pkl', self.path)
        
        
        ## ------- building CNN model -------
        model = build_model(np.array(X_train).shape[1])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
        model.fit(np.array(X_train), np.array(Y_train), validation_data=(np.array(X_val), np.array(Y_val)), epochs=self.epochs, batch_size=self.batch_size)
                
        pred = model.predict(np.array(X_test))
        evaluate = model.evaluate(np.array(X_test), np.array(Y_test))
        
        ## ------- writing model -------
        model.save(self.path+'RNN')
        write_pickle('RNN_prediction.pkl', pred, self.path)
        write_txt('RNN_test_evaluation.txt', str(evaluate), self.path)
          
        
          