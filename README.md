# Toxic Comments Classification

### About the Project: <h3>
Internet is a platform to write your opinions and the threat of abuse over the internet has increased over the years. This has stopped many people from voicing out their opinions and have a conversation over the internet. The Conversation AI team, a research initiative founded by Jigsaw and Google are working on tools to help improve online conversation. 
  
This project is about predicting the probability that a given comment is toxic. The dataset used for this project contains comment which belongs to any of the 6 classes - 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'. This is a __multi-label classification__ problem. The difference between multi class and multi label classification problem is that, in multi class there can be more than 2 classes but the sample data can belong to only one of the class. In case of multi label classification, the sample data can be classified into more than one classes.

### Text pre processing and Machine Learning: <h3>
The pre processing step of comments include removing URLs, digits, hyper links, html tags, links to image, emails, expanding contactions and removing words whose length is greater than 15. Later stemming and lemmatization was performed on tokenized comments. To convert the comments in text form into vectors, Tf-Idf Vectorizor was used with unigram feature. The dataset was divided into 70% training set and 30% test set. Various classification algorithms like Logistic Regression, XGBoost, RandomForest and NaiveBayes were used for predicting the probability of comment belonging to a particular class. Each algorithm was trained for 6 different classes. The performance of the algorithms was measure based on the F1 score. Logistic Regression with parameters n_jobs=-1 and class_weight='balanced' performed well.
  
| Algorithm      | Parameters     |
| ------------- |:-------------:| 
| XGBoost | n_estimators=150, n_jobs=-1 | 
| Logistic Regression      | n_jobs=-1, class_weight='balanced'   | 
| Random Forest | max_depth=None, n_estimators=70     |
| MultinomialNB | --- |
  
### Deployment: <h3>
The application was deployed on Heroku. Input for this application is a comment and output is the probability of the given comment belonging to each class. 

Link to Deployed Application : https://toxic-commentclassifier.herokuapp.com/

<div align="center">
  <img src="/Images/1.png" height="200" width="600"><img src="/Images/2.png" height="200" width="600">
</div>

### Installing required librarires: <h3>
* Installing nltk:
```
import nltk
nltk.download('popular')
```
* Installing xgboost:
```
pip install xgboost
```
* Installing xgboost:
```
pip install contractions
```
