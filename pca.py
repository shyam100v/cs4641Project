import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split 


# Read combined.csv file
dataset = pd.read_csv('combined.csv') 

# Split into training set and testing set
dataset_train, dataset_test = train_test_split(dataset, test_size = 0.2, random_state = 0) 

# feature scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
dataset_train = sc.fit_transform(dataset_train) 
dataset_test = sc.transform(dataset_test) 

# Applying PCA function on training and testing set 
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) 
dataset_train = pca.fit_transform(dataset_train) 
dataset_test = pca.transform(dataset_test) 

# fitting different method, then predict result