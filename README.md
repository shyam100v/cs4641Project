# What makes a YouTube video trend?

## Introduction/Background:
YouTube is an online video-sharing platform with billions of users, posting and watching content for entertainment and education. YouTube monetizes popular videos for the number of views as it increases the use and popularity of the platform itself. With our project, we hope to provide insights on how to make a video trend on YouTube and predict the popularity of a video given certain features. 


## Dataset: 
-> https://www.kaggle.com/datasnaek/youtube-new <br/>
This dataset includes several months of data on daily trending YouTube videos. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count for up to 200 trending videos per day for several regions. More information like the channel’s age, channel's video count, and subscriber count have been added using the YouTube API. From this dataset, we will only be using the USA's, Canada's, and Great Britain's trending video data. 

#### Write about YouTube API and out data collection


## Cleaning up the data
Youtube fosters content from all around the world, so we have deleted video entries in the dataset that are not in english to easier analyze the data, as not everyone in our team speaks other languages such as french. We have also removed null entries, separated the publishing time column into specific month, day, year, and time columns, and handled outliers using hard thresholds and by using the mean and standard deviation. 

### Regarding duplicates in the data
There are duplicates in the old and new data. A lot of videos in old Data did not have Video ID. They are marked as "notAvailable" in videoID. The old data had 80362 duplicates. There were totally 105094 data points in old data. 76.5% of old data were duplicates. The videos with highest number of views was retained. The new dataset had 8276 duplicates and 1483 unique values. Now totally we have 24502 (old) + 1483 (new) = 25985 unique points

### Data Format
The csv format of final file that contains both Old and New data:

1. regionTrending	
2. trendingRank	
3. timeFetched	4. videoId	
5. videoTitle	
6. videoCategoryId	
7. videoPublishTime	
8. videoDuration	
9. videoTags	
10.videoViews	
11. videoLikes	
12. videoDislikes	
13. videoCommentCount	
14. videoDescription	
15. videoLicenced	
16. channelTitle	
17. channelId	
18. channelDescription	
19. channelPublishedAt	
20. channelViewCount	
21. channelSubsCount	
22. channelVideoCount	
23. thumbnail_link	
24. comments_disabled	
25. ratings_disabled	
26. video_error_or_removed	
27. publishDateCorrectFormat	
28. trendingDateCorrectFormat	
29. dayDifference 
30. publishedZTime 
31. publishedZTimeFloat	
32. publishedDayOfWeek	
33. newOrOldData

  
Note the following for old data:
1. Added a column trendingRank and made it 0 for all rows for old data
2. Following columns are made as 0: channelViewCount, channelSubsCount, channelVideoCount, channelId
3. Following columns are made as "notAvailable": videoDuration, videoLicenced, channelDescription, channelPublishedAt
4. Correct Date format is DD-MM-YY (Sorry to the American born)

For the new data:

thumbnail_link, comments_disabled, ratings_disabled, video_error_or_removed are marked as 'notAvailable'

## Working of combiningData.ipynb

All the new data files obtained from YouTube using the API must be in a subfolder "scrapedData\" and must contain "csvOut" as part of their file name. The old data is in an excel file in the master folder (not inside any subfolders) as "finalOldData.xlsx". It is already in the correct format. The day difference between published date and the trending date is already calculated for old data. This python script generates two csv files: "newDataOnly_csv_newFormat.csv" (which contains only the new data in the correct format) and "oldAndNewData.csv" (concatenated with old data). Both the files are placed in the master folder. The script will take some time to read from "finalOldData.xlsx" and some time to write the larger output csv file. THis script does not clean the data. It only combines the two data pieces in the right format.


## Principal Component Analysis(PCA)
We selected some properties from original video dataset as features of videos, including trending rank, video category, number of views, likes, and dislikes, number of comments, publish time, and video channel related features. For features like duration of video and publish time, we preprocessed our data such that they are represented in the same unit(second) and in a 24-hour time scale.
We combined all 12 features into a training dataset and apply PCA. PCA was used to reduce the dimension of features through capturing variation. Here is the cumulative explained variance plot. 

![pca_variance](https://github.com/shyam100v/cs4641Project/blob/master/image/pca_variance.PNG)

From the plot, we can see that at 6 components, we will get a desired cumulative explained variance(0.9). We also made two component PCA scatter plots to give us some visualizations.

![pca_views](https://github.com/shyam100v/cs4641Project/blob/master/image/pca_views.PNG)
![pca_likes](https://github.com/shyam100v/cs4641Project/blob/master/image/pca_likes.PNG)
![pca_dislikes](https://github.com/shyam100v/cs4641Project/blob/master/image/pca_dislikes.PNG)

From the scatter plots, we do can see number of views, likes, and dislikes are correlated. Low number of views, low number of likes, and low number of dislikes are all clustered at the middle-left part of the graph.

## DBSCAN
**Publishing Times**\
Using DBSCAN clustering on the video views and publishing time features, we can see that the optimal time frame to publish videos on YouTube is from about 1:30 to 8:30 pm GMT; however, we did find many noise points and the clusters found were quite low in view count. 

**Video Titles**\
One of our main expected outcomes for the overall analysis of youtube videos was that video titles with similar characteristics to clickbait titles would have a higher view count. The main quantifiable characteristics that we seemed to find in all clickbait titles were capital letters and punctuation marks such as exlcamations points and question marks. We also figured that the length of the title could play a role as longer titles can give more information on the context of the videos. Using these three characteristics as a starting point, we manipulated our data to quanitfy these characteristics and then used DBSCAN to cluster them. Hopefully, we would be able to see clear clusters and relationships between the characteristics and the number of views. The below graphs are the results.

![title length dbscan](https://github.com/shyam100v/cs4641Project/blob/master/image/Length_of_Title_DBSCAN.png)

![capital letters dbscan](https://github.com/shyam100v/cs4641Project/blob/master/image/Num_Caps_DBSCAN.png)

![punctuation dbscan](https://github.com/shyam100v/cs4641Project/blob/master/image/Num_Puncs_DBSCAN.png)

In the graph, the darkest purple points represent the outliers or, in other words, the points that did not fit into any cluster. We can see here that there actually a much smaller correlation between the view count of a video and its title's characteristics. The largest clusters in the first two graphs show that there is practically no relation between the length of the title or the number of capital letters in the title and the view count. But, when looking at the clusters at a higher view count, we can actually see a little of the opposite of our expected outcome. Instead of long video titles, we can see higher view count clusters aroung the midpoint in video titles and instead of having a lot of capital letters, we actually see clusters with less capital letters have a higher view count. Finally, with the punctuation marks, we can see that clusters with high view counts actually have on average little to no punctuation marks. 

To conlcude our findings, we can see that the majority of videos do not rely on the characteristics of the video titles. For those that do have some correlation, we can see that the findings are the opposite of what we originally expected to see.




## Linear Regression
First, an analysis of the correlation between different variables was performed. Looking at the first row, it is seen that only likes, dislikes and comment count are correlated with the number of views of a video.
![Correlation table](https://github.com/shyam100v/cs4641Project/blob/master/image/Correlation%20table.PNG)

Then, a statistical summary of each of our variables was obtained. This analysis is useful to potentially identify any outliers in the data.
![max-min-std](https://github.com/shyam100v/cs4641Project/blob/master/image/max-min-std.PNG)

Based on the table above, the data for the number of views, likes and dislikes was modified to remove any data points that were outside 2 standard deviations from their respective mean. Then, a basic histogram of the number of views, likes and dislikes was plotted. As it is appreciated on the graphs below, the three graphs are heavily skewed which is understandable — most common YouTubers probably won’t have that many views, likes and dislikes. Ideally, the data should resemble a Gaussian distribution. Luckily, a log tranformation can be applied to the number of views, likes and dislikes to achieve that.

![Number of views without logs](https://github.com/shyam100v/cs4641Project/blob/master/image/Number%20of%20views%20without%20logs.PNG)
![Number of dislikes without logs](https://github.com/shyam100v/cs4641Project/blob/master/image/Number%20of%20dislikes%20without%20logs.PNG)
![Number of likes without logs](https://github.com/shyam100v/cs4641Project/blob/master/image/Number%20of%20likes%20without%20logs.PNG)
![Log number of dislikes](https://github.com/shyam100v/cs4641Project/blob/master/image/Log%20number%20of%20dislikes.PNG)
![Log number of likes](https://github.com/shyam100v/cs4641Project/blob/master/image/Log%20number%20of%20likes.PNG)
![Log number of views](https://github.com/shyam100v/cs4641Project/blob/master/image/Log%20number%20of%20views.PNG)

Using this 3 histograms above, a linear regression of the log number of views versus the log number of likes and dislikes can be plotted. As expected, both correlations show an R^2 greater than 0.6, showing a big correlation between both the number of likes and dislikes and the number of views of a video. This analysis was only performed on these 2 variables since no other variables appear to correlate with the number of views of a video

![Linear regression of log views versus log dislikes](https://github.com/shyam100v/cs4641Project/blob/master/image/Linear%20regression%20of%20log%20views%20versus%20log%20dislikes.PNG)
![Linear regression of log views versus log likes](https://github.com/shyam100v/cs4641Project/blob/master/image/Linear%20regression%20of%20log%20views%20versus%20log%20likes.PNG)

## Multiple Regression
As we are trying to aid users in creating a popular Youtube video, we will look at one of the main features (excluding actual video content) that can manipulated by the video maker; the video title. Via Multiple Regression, we will use the quantifiable features of a video title to create a model that can predict the number of views a video will recieve based on it's title. In this case we will be looking at the length, number of capital letters, and number of exclamation and questions marks in the video title. Once again, we chose these parameters as they are common characteristics of clickbait titles, which are made exclusively for people to be attracted to.

First we loaded our data and calculated the needed parameters from the given titles. Now, a major hinderance with our original data was that there were a few drastic outliers in relation to the number of views a video had. In order to eliminate those, we removed any data points that were more than one standard deviation away from the mean (which was a very large standard deviation, so it was fitting to only use one). But even then, our data was very skewed. In order to have a balanced model, we would ideally like a Gaussian distribution, so we decided to take the log of the number of views, which did in fact give us a Gausian ditribution. The below graphs show the data distribution before and after applying the log function. 

![Views before log function](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_viewskew.png)

![Views after log function](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_viewlog.png)

Now that our data is better suited for analysis, we moved on to actually applying the regression model. Since we used three parameters (length, number of capital letters, and number of exclamation and queston marks) versus the number of views on a video, it would have resulted in a 3-dimensional model in a 4-dimensional space. As it is difficult to understand a 4-dimensional model, the following graphs show the regression models of just two parameters each versus the log of the number of views of the videos in order to give an overall picture.

![length vs caps](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_lengthcapsreg.png)

![length vs puncs](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_lengthpuncsreg.png)

![caps vs puncs](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_capspuncsreg.png)

As can be seen above, the grey plane in each of the graphs is the regression model that fits the set of data points the best. Since our data points are fairly spread out and do not show much correlation to the number of views, the fitted planes are relatively flat, and center ariund the area where most of the data points are in order to at least fit for a majority of videos. 

Finally, to see how well our prediction model actually worked, we graphed a random set of data points that were used in our testing phase and compared them to their predictions.

![predicted vs actual](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_predvsact.png)

As we can see here, the model tends to predict within the 200000 to 400000 views range, and this is probably due to the very flat model that can be seen in the previous graphs. Though the model seems to overestimate the number of views for anything below 300000 views, it is actually proportional in a sense and predicts slighty less for those with less views and predicts slightly more for those with more views. Now, since it can be seen that the videos with a high number of views have very low predictions, we also calculated the error of the preditcions of videos with different ranges of views.

![accuracy](https://github.com/shyam100v/cs4641Project/blob/master/image/MultReg_accuracy.png)

In the above graph, it can be seen that the lower the actual number of views a video has, the more accurate this model will be in its predictions. 

In conclusion, we see that the characteristics of a video title actually have a much smaller effect on the popularity of a video than we originally believed, especially on videos with extremely high view count. This is why this multiple regression model was so flat, because majority of videos actually have a relatively low view count no matter how they construct their video titles.


## Gradient Boosting Regressor

Gradient Boosting Regressor is a form of tree ensemble model which builds an ensemble of weak predistion models. A new tree is trained at each step additively over the previous stage with the loss function as the residual error from previous stage. The [sklearn.ensemble.GradientBoostingRegressor model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) from sklearn module is used in this analysis. The data collected using the YouTube API is used to train and test the model with the number of views of a video as the target label. As stated in the above sections, the logarithm of the number of views is closer to a normal distribution and is hence used in place of the absolute value of number of views for training and testing.

__Data preparation__: 
The collected data is first filtered to exclude outliers in the number of views. For this analysis, only the videos that have a view count between 1000 and 10 million are used. The logarithm of the number of views, likes and dislikes is shown below. It is seen that they roughly follow a normal distribution. 

MDI feature importance


## Our Analysis and Insights
 The following is a summary of the key results and insights from the analyses that we carried out:

## Future Work 

The following are the possible directions in which we would have enhanced our work if we had more time:



## References:
[1] Gill, Phillipa et al. "Youtube traffic characterization: a view from the edge." Proceedings of the 7th ACM SIGCOMM conference on Internet measurement. 2007. <br/>
[2] F Figueiredoet al "The tube over time: characterizing popularity growth of youtube videos." Proceedings of the fourth ACM international conference on Web search and data mining. 2011.<br/>
[3] G. Chatzopoulou et al, "A First Step Towards Understanding Popularity in YouTube," INFOCOM IEEE Conference on Computer Communications Workshops, CA, 2010 <br/>
[4] Coding, Sigma. “How to Build a Linear Regression Model in Python | Part 1.” Youtube, Apr. 2019, www.youtube.com/watch?v=MRm5sBfdBBQ.



