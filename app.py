# Import libraries
import numpy as np
import pandas as pd
import re
from flask import Flask, request, jsonify
from googletrans import Translator
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

#model code
#Importing the data
data = pd.read_csv("data.csv", usecols = ["Crop", "QueryType", "QueryText", "KccAns"])
#print(data.head())

#PreProcess Data
df = pd.DataFrame(columns=['Crop', 'QueryType', 'QueryText', 'KccAns'])
for index, row in data.iterrows():
    b = []
    crop = row['Crop']
    a = crop.split('(')
    for x in a:
        temp_list = x.split('/')
        for crop in temp_list:
            crop.lower()
            b.append(crop)
    b[:] = [s.lower() for s in b]
    b[:] = [s.strip(')') for s in b]
    b[:] = [s.strip() for s in b]
    for x in b:
        df = df.append({'Crop':x, 'QueryType': row['QueryType'], 'QueryText': row['QueryText'], 'KccAns': row['KccAns']}, ignore_index=True)
data = df
print(type(data))

#Test-Train split
X_train, X_test, y_train, y_test = train_test_split(data['QueryText'], data['QueryType'], test_size=0.33, random_state=53)

#Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)

#Tfidf Transformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(count_train)

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api', methods=['POST'])
def predict():
    dt = request.get_json(force=True)
    print('OUTPUT:')
    input = dt['exp']
    input = [input]
    l= Translator().detect(input[0]).lang
    input[0] = Translator().translate(input[0]).text
    input_count = count_vectorizer.transform(input)
    input_tfidf = tfidf_transformer.transform(input_count)
    prediction = model.predict(input_tfidf)
    output = prediction[0]
    
    test=[]
    for j in data['Crop']:
        test.append(str(j))
    entity = ""
    for w in input[0].split():
        if re.match(w.lower(),'purple'):
            w = 'brinjal'
        for c in test:
            if re.match(w.lower(),c.lower()):
                entity=c

    for i in range(len(data)):
        crp = data.loc[i, 'Crop']
        ent = data.loc[i, 'QueryType']
        if crp == entity and ent == output:
            r = data.loc[i,'KccAns']
        elif ent == output and crp!=entity:
            r = data.loc[i,'KccAns']
    resp = Translator().translate(r,dest = l).text
    print('Query: ', input)    
    print('Answer: ', resp.encode())
    return resp.encode()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
