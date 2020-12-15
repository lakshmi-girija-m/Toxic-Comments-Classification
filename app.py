from flask import Flask, request, render_template
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
import re
import contractions

app = Flask(__name__)

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def text_preprocess(sent):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer() 
    
    # convert all words to lowercase
    sent = sent.lower()
    
    # expand contractions
    expanded_words = []
    for word in sent.split():
        if len(word)>20:
            continue
        else:
            expanded_words.append(contractions.fix(word))
    sent = ' '.join(expanded_words) 
    
    # remove html tags
    sent = re.sub('{html}', "", sent)
    
    # remove http links and web site url
    sent = re.sub(r"http\S+", "", sent)
    sent = re.sub(r"www\S+", "", sent)
    
    # remove numbers
    sent = re.sub('[0-9]+', '', sent)
    
    # remove words utc and image
    sent = sent.replace('utc', '')
    
    # remove words with .jpg
    sent = re.sub(r"[a-zA-Z]*[0-9]*.jpg", '', sent)
    
    # remove emails
    sent = re.sub(r"\S*@\S*\s?", '', sent)
    
    # tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    
    # remove stop words
    filtered_words = [w for w in tokens if w not in stopwords]
    
    stem_words = [stemmer.stem(w) for w in filtered_words]
    lemma_words = [lemmatizer.lemmatize(w) for w in stem_words]
    
    return " ".join(lemma_words)
 
with open('tfidf.pkl', "rb") as f:
    tfidf = pickle.load(f)

with open('toxic.pkl', "rb") as f:
    toxic = pickle.load(f)
with open('severe_toxic.pkl', "rb") as f:
    severe_toxic = pickle.load(f)
with open('obscene.pkl', "rb") as f:
    obscene = pickle.load(f)
with open('threat.pkl', "rb") as f:
    threat = pickle.load(f)
with open('insult.pkl', "rb") as f:
    insult = pickle.load(f)
with open('identity_hate.pkl', "rb") as f:
    identity_hate = pickle.load(f)


@app.route('/')
def home():
   return render_template('index.html')
            
@app.route('/predict', methods=['POST'])
def predict():
   comment = [text_preprocess(str(x)) for x in request.form.values()]
   data = tfidf.transform(comment)
   toxic_pred = toxic.predict_proba(data)
   severe_toxic_pred = severe_toxic.predict_proba(data)
   obscene_pred = obscene.predict_proba(data)
   threat_pred = threat.predict_proba(data)
   insult_pred = insult.predict_proba(data)
   identity_hate_pred = identity_hate.predict_proba(data)
   
   labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
   results = [toxic_pred[0][1], severe_toxic_pred[0][1], obscene_pred[0][1], threat_pred[0][1], insult_pred[0][1], identity_hate_pred[0][1]]
   
   #final_results = list(zip(labels, results))
   
   return render_template('index.html', labels=labels, results=results)

if __name__ == "__main__":
   app.run(debug=True)