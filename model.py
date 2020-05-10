import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

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

#Test-Train split
X_train, X_test, y_train, y_test = train_test_split(data['QueryText'], data['QueryType'], test_size=0.33, random_state=53)

#Count Vectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)

#Tfidf Transformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(count_train)
X_test_tfidf = tfidf_transformer.transform(count_test)

#MultinomialNB (accuracy: 0.5596330275229358)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test_tfidf)
np.mean(predicted==y_test)

#SGD Classifier (accuracy: 0.6880733944954128)
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)
clf_sgd = sgd.fit(X_train_tfidf, y_train)
#predicted_sgd = clf_sgd.predict(X_test_tfidf)
#print('Accuracy: ', np.mean(predicted_sgd == y_test))
pickle.dump(clf_sgd, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict(X_test_tfidf))

#MLP Classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 50, 70), random_state=1)
clf.fit(X_train_tfidf, y_train)
pred = clf.predict(X_test_tfidf)
np.mean(pred == y_test)

#extracting the entities

def getIntent(input):
    input_count = count_vectorizer.transform(input)
    input_tfidf = tfidf_transformer.transform(input_count)
    print("prediction: ", clf_sgd.predict(input_tfidf))

def getEntity(input):
    entity = "others"
    words = input.split();
    
