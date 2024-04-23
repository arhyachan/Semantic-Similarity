from flask import Flask, jsonify, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import gensim
import regex as re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import scipy


app = Flask(__name__)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lem = WordNetLemmatizer()
sw = set(stopwords.words('english'))

def preprocess(s):
  
  tokens = word_tokenize(s)
  # print(tokens)

  filteredtokens = []
  for w in tokens:
      sm = re.sub('[^A-Za-z]',' ', str(w))
      # x = re.split("\s", sm)
      filteredtokens.append(sm)
  # print(filteredtokens)

  for sent in filteredtokens:
    if sent == ' ':
      filteredtokens.remove(sent)
  # print(filteredtokens)

  lowercased = []
  for i in filteredtokens:
      i = i.lower()
      lowercased.append(i)
  # print("lowercased", lowercased)

  lemma = []
  for sent in lowercased:
      token = lem.lemmatize(sent)
      lemma.append(token)
  # print(lemma)

  clean_tokens = [w for w in lemma if not w in sw]
  # clean_tokens = []
  return clean_tokens


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_similarity', methods=['POST'])
def calculate():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        
        # Preprocess the text inputs
        preprocessed_text1 = preprocess(text1)
        preprocessed_text2 = preprocess(text2)
        
        model = Doc2Vec.load("semanticsim_d2v.model")

        s1_vector = model.infer_vector(preprocessed_text1)
        s2_vector = model.infer_vector(preprocessed_text2)
        cosine = scipy.spatial.distance.cosine(s1_vector, s2_vector)
        similarity = (1 - cosine)
        
        return jsonify(similarity_score=similarity,text1=text1,text2=text2)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
