import pandas as pd 
import numpy as np 
news_df = pd.read_csv("C:/Users/Setu Suyagya/Downloads/Data Sets/news.csv",sep=",") 
import string 
news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 }) 
news_df= news_df.replace(np.nan, '', regex=True) 
news_df.tail(6) 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(news_df['TITLE'],news_df['CATEGORY'],random_state = 1) 
print("Training dataset: ", X_train.shape[0]) 
print("Test dataset: ", X_test.shape[0])
from sklearn.feature_extraction.text import CountVectorizer 
count_vector = CountVectorizer(stop_words = "english") 
training_data = count_vector.fit_transform(X_train) 
testing_data = count_vector.transform(X_test) 
from sklearn.naive_bayes import MultinomialNB 
naive_bayes = MultinomialNB() 
naive_bayes.fit(training_data, y_train) 
predictions = naive_bayes.predict(testing_data) 
predictions 
from sklearn.metrics import accuracy_score 
print("Accuracy:", accuracy_score(y_test, predictions)) 
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix 
print("confusion matrix is \n", confusion_matrix(y_test,predictions))
