import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics


#1 Read in the dataset and initizlize a data_frame
df = pd.read_csv("./smsspamcollection.tsv", sep="\t")

#2 Split the dataset into train and test
X = df["message"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#3 Create a pipe line to vectorize the X_train feature and also create a classifier
text_clf = Pipeline([("tfidf", TfidfVectorizer()),("clf",LinearSVC())])
text_clf.fit(X_train,y_train)

#4 Evalute the model
predictions = text_clf.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

#5 Deploying the model
print(text_clf.predict(["Hello Huzaifa, kiase hu?"]))