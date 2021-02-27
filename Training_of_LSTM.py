#!/usr/bin/env python3
#Imports : 
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow import keras
import pickle
import pyarrow.parquet as pq
from gensim.models import Word2Vec
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn import preprocessing
##### REQUIRES THE DATAFRAME FOLDER TO BE NAMED 'Cohorts', WHICH INCLUDES ALL PRECOMPUTED DATAFRAMES #####
import time
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import reduce
from ppca import PPCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import json
from sklearn import metrics
from sklearn.decomposition import FastICA
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from pickle import load
from pickle import dump
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc
import scipy.stats as stats
import researchpy as rp
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split  
from keras.layers import Input, Dense 
from keras.models import Model, Sequential 
from keras import regularizers 
import umap
from sklearn.cluster import DBSCAN
import hdbscan
from statistics import mean 

print('imports Done ')
#Parameters Dataset
per_day=False
embedding_method='cbow'
embedding_size='20'

# embedded sequence function: 
#Timeseries_per_patient_LSTM_Data_embedded_cbow_dim20_win5_mc0.txt
def load_dataset(per_day,embedding_method,embedding_size):
    if per_day: 
        df_name='Timeseries_per_patient_per_day_LSTM_Data_embedded_'
    else: 
        df_name='Timeseries_per_patient_LSTM_Data_embedded_'
    if embedding_method=='cbow':
        df_name=df_name+'cbow_dim{}_win5_mc0'.format(embedding_size)
    if embedding_method=='skipgram':
        df_name=df_name+'skipgram_dim{}_win5_mc0'.format(embedding_size)
    print(df_name)
    with open("Cohort/Time_Series/"+df_name+'.txt', "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    data_sample= data[:300]
    return data,data_sample, df_name

#Load the DataSet: 
#configure dataset that should be used 
data,sample,df_name=load_dataset(per_day,embedding_method,embedding_size)

#parameters model: 
data_structure='per_patient'#'per_patient_per_day'
timesteps=len(data[0])
n_features=len(data[0][0])
layer_size_1=32
layer_size_2=16
activation_func='tanh'
optimizer_func='adam'
loss_func='mse'
n_epochs=3
n_batch_size=100
X=sample
print('Config Done ')
#define the model: 
model = Sequential()
model.add(LSTM(layer_size_1, activation=activation_func, input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(layer_size_2, activation=activation_func, return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(layer_size_2, activation=activation_func, return_sequences=True))
model.add(LSTM(layer_size_1, activation=activation_func, return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer=optimizer_func, loss=loss_func,metrics=[loss_func])
model.summary()

#fit the model : 
hist=model.fit(X, X, epochs=n_epochs, batch_size=n_batch_size, verbose=1)
# demonstrate reconstruction
#print(history.history['val_loss'][(epochs-1)])
yhat = model.predict(X, verbose=0)
print('Test Model Name ')
print('---Predicted---')
print(np.round(yhat,3))
print('---Actual---')
print(np.round(X, 3))


#get the loss 
loss_value=hist.history[loss_func][n_epochs-1]

# save the model statistics: 
result=pq.read_table('Cohort/Metrics_LSTM.parquet').to_pandas()
result=result.append({'data_structure':data_structure,'timesteps':timesteps,'n_features':n_features,'layer_size_1':layer_size_1,'layer_size_2':layer_size_2,'activation_func':activation_func,'optimizer_func':optimizer_func,'loss_func':loss_func,'n_epochs':n_epochs,'n_batch_size':n_batch_size,'loss':loss_value}, ignore_index=True)
result.to_parquet('Cohort/Metrics_LSTM.parquet')
result