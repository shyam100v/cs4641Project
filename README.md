# cs4641Project

## Introduction/Background:
YouTube is an online video-sharing platform with billions of users, posting and watching content for entertainment and education. YouTube monetizes popular videos for the number of views as it increases the use and popularity of the platform itself. With our project, we hope to provide insights on how to make a video trend on YouTube and predict the popularity of a video given certain features. 


## Dataset: 
-> https://www.kaggle.com/datasnaek/youtube-new <br/>
This dataset includes several months of data on daily trending YouTube videos. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count for up to 200 trending videos per day for several regions. More information like the channel’s age, channel's video count, and subscriber count have been added using the YouTube API.


# Cleaning up the data
-> Youtube fosters content from all around the world, so we have deleted video entries in the dataset that are not in english to easier analyze the data, as not everyone in our team speaks other languages such as french. 
-> We removed null entries
-> We separated the publishing time column into specific month, day, year, and time columns.
-> We handled outliers using hard thresholds and by using the mean and standard deviation. 

## Regarding duplicates in the data
There are duplicates in the old and new data. A lot of videos in old Data did not have Video ID. They are marked as "notAvailable" in videoID. The old data had 80362 duplicates. There were totally 105094 data points in old data. 76.5% of old data were duplicates. The videos with highest number of views was retained. The new dataset had 8276 duplicates and 1483 unique values. Now totally we have 24502 (old) + 1483 (new) = 25985 unique points

## Data Format
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

