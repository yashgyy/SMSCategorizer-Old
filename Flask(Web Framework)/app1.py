from flask import Flask,request
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
app=Flask(__name__)
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
@app.route('/',methods=['GET'])
def Print():
    classifier=joblib.load('NLTK[RANDOMFOREST].pkl')
    return String1,220
@app.route('/',methods=['POST'])
def print1():
    return "Hello1",200
if  __name__=="__main__":
    classifier=joblib.load('NLTK[RANDOMFOREST].pkl')
    String1=classifier.predict(['Hello'])
    String1=String1[0]
    print(String1)
    app.run(port=8000,use_reloader=True)


