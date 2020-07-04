# cs4641Project

## Cleaning up the data

Youtube fosters content from all around the world, so we have deleted video entries in the dataset that are not in english to easier analyze the data, as not everyone in our team speaks other languages such as french. 

## Data Format

The csv format of final file that contains both Old and New data:

{regionTrending	trendingRank	timeFetched	videoId	videoTitle	videoCategoryId	videoPublishTime	videoDuration	videoTags	videoViews	videoLikes	videoDislikes	videoCommentCount	videoDescription	videoLicenced	channelTitle	channelId	channelDescription	channelPublishedAt	channelViewCount	channelSubsCount	channelVideoCount	thumbnail_link	comments_disabled	ratings_disabled	video_error_or_removed	publishDateCorrectFormat	trendingDateCorrectFOrmat	dayDifference	newOrOldData}
  
Note the following for old data:
1. Added a column trendingRank and made it 0 for all rows for old data
2. Following columns are made as 0: channelViewCount, channelSubsCount, channelVideoCount, channelId
3. Following columns are made as "notAvailable": videoDuration, videoLicenced, channelDescription, channelPublishedAt
4. Correct Date format is DD-MM-YY (Sorry to the American born)

For the new data:

thumbnail_link, comments_disabled, ratings_disabled, video_error_or_removed are marked as 'notAvailable'

## Working of combiningData.ipynb

All the new data files obtained from YouTube using the API must be in a subfolder "scrapedData\" and must contain "csvOut" as part of their file name. The old data is in an excel file in the master folder (not inside any subfolders) as "finalOldData.xlsx". It is already in the correct format. The day difference between published date and the trending date is already calculated for old data. This python script generates two csv files: "newDataOnly_csv_newFormat.csv" (which contains only the new data in the correct format) and "oldAndNewData.csv" (concatenated with old data). Both the files are placed in the master folder. The script will take some time to read from "finalOldData.xlsx" and some time to write the larger output csv file. THis script does not clean the data. It only combines the two data pieces in the right format.

