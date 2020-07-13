# What makes a YouTube video trend?

## Introduction/Background:
YouTube is an online video-sharing platform with billions of users, posting and watching content for entertainment and education. YouTube monetizes popular videos for the number of views as it increases the use and popularity of the platform itself. With our project, we hope to provide insights on how to make a video trend on YouTube and predict the popularity of a video given certain features. 


## Dataset: 
-> https://www.kaggle.com/datasnaek/youtube-new <br/>
This dataset includes several months of data on daily trending YouTube videos. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count for up to 200 trending videos per day for several regions. More information like the channel’s age, channel's video count, and subscriber count have been added using the YouTube API. From this dataset, we will only be using the USA's, Canada's, and Great Britain's trending video data. 


## Cleaning up the data
Youtube fosters content from all around the world, so we have deleted video entries in the dataset that are not in english to easier analyze the data, as not everyone in our team speaks other languages such as french. We have also removed null entries, separated the publishing time column into specific month, day, year, and time columns, and handled outliers using hard thresholds and by using the mean and standard deviation. 

### Regarding duplicates in the data
There are duplicates in the old and new data. A lot of videos in old Data did not have Video ID. They are marked as "notAvailable" in videoID. The old data had 80362 duplicates. There were totally 105094 data points in old data. 76.5% of old data were duplicates. The videos with highest number of views was retained. The new dataset had 8276 duplicates and 1483 unique values. Now totally we have 24502 (old) + 1483 (new) = 25985 unique points

### Data Format
The csv format of final file that contains both Old and New data:
|                     |                             |                               | 
| ------------------- | --------------------------- | ----------------------------- | 
|1. regionTrending    |   12. videoDislikes         |  23. thumbnail_link           | 
|2. trendingRank	    |   13. videoCommentCount	    |  24. comments_disabled        |
|3. timeFetched	      |   14. videoDescription      |  25. ratings_disabled	        | 
|4. videoId	          |   15. videoLicenced	        |  26. video_error_or_removed   |
|5. videoTitle	      |   16. channelTitle          |  27. publishDateCorrectFormat	|
|6. videoCategoryId	  |   17. channelId	            |  28. trendingDateCorrectFormat|	
|7. videoPublishTime	|   18. channelDescription	  |  29. dayDifference            |
|8. videoDuration	    |   19. channelPublishedAt	  |  30. publishedZTime           |
|9. videoTags	        |   20. channelViewCount      |  31. publishedZTimeFloat	    | 
|10. videoViews	      |   21. channelSubsCount	    |  32. publishedDayOfWeek	      |
|11. videoLikes	      |   22. channelVideoCount	    |  33. newOrOldData             |



| Left-aligned        | Center-aligned         | Right-aligned              |
| :---                |     :---:              |                       ---: |
| 1. regionTrending   | 12. videoDislikes      | 23. thumbnail_link	    |
| 2. trendingRank     | 13. videoCommentCount  | 24. comments_disabled	    |

  
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
Using DBSCAN clustering on the video views and publishing time features, we can see that the optimal time frame to publish videos on YouTube is from about 1:30 to 8:30 pm GMT; however, we did find many noise points and the clusters found were quite low in view count. 


## GradientBoostingRegressor




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


## Our Analysis and Insights
 do we need this section if we already mention the insights in each section above?

## Future Work (not sure what this section is for)

## References:
[1] Gill, Phillipa et al. "Youtube traffic characterization: a view from the edge." Proceedings of the 7th ACM SIGCOMM conference on Internet measurement. 2007. <br/>
[2] F Figueiredoet al "The tube over time: characterizing popularity growth of youtube videos." Proceedings of the fourth ACM international conference on Web search and data mining. 2011.<br/>
[3] G. Chatzopoulou et al, "A First Step Towards Understanding Popularity in YouTube," INFOCOM IEEE Conference on Computer Communications Workshops, CA, 2010 <br/>
[4] Coding, Sigma. “How to Build a Linear Regression Model in Python | Part 1.” Youtube, Apr. 2019, www.youtube.com/watch?v=MRm5sBfdBBQ.



