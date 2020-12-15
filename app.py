from flask import Flask, request, render_template
import pickle
import re
import contractions

app = Flask(__name__)

def text_preprocess(sent):
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
    
    # removing punctuations
    sent = re.sub(r'[^\w\s]', '', sent) 
    
    return sent
 
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
   
   return render_template('index.html', labels=labels, results=results)

if __name__ == "__main__":
   app.run(debug=True)
