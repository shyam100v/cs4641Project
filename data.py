import csv

fieldnames = ["numberOfCharactersInTitle", "numberOfCharactersInDescription", 
"uppercaseCharactersInTitle", "uppercaseCharactersInDescription", "numberOfWordsInTitle", "numberOf!InTitle"]
charArr1 = []
charArr2 = []
upperArr1 = []
upperArr2 = []
numWordArr = []
numPMArr = []
with open('combined_csv.csv', encoding='utf-8', mode = 'r', newline='') as myFile:
    reader = csv.DictReader(myFile)
    for row in reader:
        charArr1.append(len(row["videoTitle"]))
        charArr2.append(len(row["videoDescription"]))
        upperArr1.append(sum(1 for c in row["videoTitle"] if c.isupper()))
        upperArr2.append(sum(1 for c in row["videoDescription"] if c.isupper()))
        numWordArr.append(len(row["videoTitle"].split()))
        numPMArr.append(len(row["videoTitle"]) - len(row["videoTitle"].rstrip('!')))

with open('after.csv', encoding='utf-8', mode = 'w', newline='') as myFile:
    writer = csv.DictWriter(myFile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(charArr1)):
        writer.writerow({"numberOfCharactersInTitle": charArr1[i], "numberOfCharactersInDescription": charArr2[i],
        "uppercaseCharactersInTitle": upperArr1[i], "uppercaseCharactersInDescription": upperArr2[i],
        "numberOfWordsInTitle": numWordArr[i], "numberOf!InTitle": numPMArr[i]})

