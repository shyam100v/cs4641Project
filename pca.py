import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Read combined.csv file
columns = ["videoCategoryId", "videoViews", "videoLikes", "videoDislikes", "videoCommentCount", "publishedZTimeFloat", "publishedDayOfWeek"]
dataset = pd.read_csv('oldAndNewOnlyEnglish_noDuplicates.csv', usecols = columns) 
videoLikes = dataset["videoLikes"]
videoLikes.values[videoLikes > 10000] = 1
videoLikes.values[videoLikes > 1] = 0
# dataset.head()
# print(len(videoLikes))

# Standardize data
features = ["videoCategoryId", "videoViews", "publishedZTimeFloat", "publishedDayOfWeek"]
x = dataset.loc[:, features].values
y = dataset.loc[:,["videoLikes"]].values
x = StandardScaler().fit_transform(x)

# Applying PCA 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dataset["videoLikes"]], axis = 1)
finalDf.head(5)

# plot data
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targetLabel = ['High number of likes', 'Low number of likes']
targets = [1, 0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf["videoLikes"] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targetLabel)
ax.grid()