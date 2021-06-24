import numpy as np
import pandas as pd 
from keras.models import model_from_json


embeddings = {}
with open('glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs
    
def getOutputEmbeddings(X):
    
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
            
    return embedding_matrix_output

with open("services\emojifier\model.json", "r") as file:
    model = model_from_json(file.read())
model.load_weights("services\emojifier\model.h5")
model._make_predict_function()

def predict(x):
    X = pd.Series([test_str])
    emb_X = getOutputEmbeddings(X)
    p = model.predict_classes(emb_X)
    return emoji.emojize(emoji_dictionary[str(p[0])])