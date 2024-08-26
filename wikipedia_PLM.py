import csv
from mediawiki import MediaWiki # pymediawiki 0.7.4
wiki = MediaWiki()
"""
Product lifecycle
Business analysis
Axiomatic product development lifecycle
"""
def downloadWikipedia(x):
    p=wiki.page(x, auto_suggest=False)
    f=open(p.title, 'wb')
    f.write(p.content.encode())
    f.close()

def readCSV(y_col):# 1 2
    csv_file=open("dataset.csv", "r") # відкрити файл для читання
    reader=csv.reader(csv_file,delimiter = ';') # об'єкт для читання
    X,Y=[],[]
    for row in reader:
        if row[y_col]=='':continue
        X.append(row[0])
        Y.append(row[y_col])
    csv_file.close() # закрити файл
    return X,Y

X_,Y=readCSV(1)
X=[]
for x in X_:
    try:
        f=open(x, 'rb')
        X.append(f.read())
        f.close()
    except FileNotFoundError:
        print("download: ", x)
        downloadWikipedia(x)

class Dataset:
    pass
dataset=Dataset()
dataset.data=X
dataset.target=Y