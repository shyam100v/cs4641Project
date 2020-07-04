# cs4641Project

## Cleaning up the data

Youtube fosters content from all around the world, so we have deleted video entries in the dataset that are not in english to easier analyze the data, as not everyone in our team speaks other languages such as french. 

## Working of combiningData.ipynb

All the new data files obtained from YouTube using the API must be in a subfolder "scrapedData\" and must contain "csvOut" as part of their file name. The old data is in an excel file in the master folder (not inside any subfolders) as "finalOldData.xlsx". It is already in the correct format. The day difference between published date and the trending date is already calculated for old data. This python script generates two csv files: "newDataOnly_csv_newFormat.csv" (which contains only the new data in the correct format) and "oldAndNewData.csv" (concatenated with old data). Both the files are placed in the master folder. The script will take some time to read from "finalOldData.xlsx" and some time to write the larger output csv file. This script does not clean the data. It only combines the two data pieces in the right format.

