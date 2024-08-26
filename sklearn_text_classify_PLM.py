#https://reintech.io/blog/how-to-create-a-text-classification-model-with-scikit-learn
import numpy as np
import sklearn
import wikipedia_PLM
dataset=wikipedia_PLM.dataset
dataset.target=np.array(dataset.target)

print("Number of documents: ", len(dataset.data))
#print("Number of categories: ", len(dataset.target_names))
from sklearn.feature_extraction.text import CountVectorizer#, TfidfVectorizer
from sklearn.feature_extraction import _stop_words
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

vectorizer = CountVectorizer(stop_words='english', analyzer=stemmed_words)
X = vectorizer.fit_transform(dataset.data)
y = dataset.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
#from bert_sklearn import BertClassifier # потребує .toarray()
#https://github.com/charles9n/bert-sklearn
clf=MultinomialNB()
"""
clf = BertClassifier()
clf.bert_model = 'bert-large-uncased'
clf.num_mlp_layers = 3
clf.max_seq_length = 196
clf.epochs = 4
clf.learning_rate = 4e-5
clf.gradient_accumulation_steps = 4
"""
clf.fit(X_train.toarray(), y_train)
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test.toarray())
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

myX = vectorizer.transform(["In solid modeling and computer-aided design, boundary representation (often abbreviated B-rep or BREP) is a method for representing a 3D shape[1] by defining the limits of its volume. A solid is represented as a collection of connected surface elements, which define the boundary between interior and exterior points."]).toarray()
print(clf.predict(myX))
print(clf.predict_proba(myX))

myX = vectorizer.transform(["Lean Six Sigma is a process improvement approach that uses a collaborative team effort to improve performance by systematically removing operational waste[1] and reducing process variation. It combines Lean Management and Six Sigma to increase the velocity of value creation in business processes. "]).toarray()
print(clf.predict(myX))
print(clf.predict_proba(myX))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # матриця помилок

"""
# будуємо криві навчання для моделі
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, train_sizes=np.linspace(0.2, 1, 20))
plt.plot(train_sizes, np.mean(train_scores, 1), 'o-')# оцінка навчання
plt.plot(train_sizes, np.mean(test_scores, 1), 'o--')# оцінка перевірки
plt.xlabel('train_size'),plt.ylabel('score')
plt.grid()
plt.show()
"""