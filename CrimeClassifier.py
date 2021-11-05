from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import codecs
import pandas as pn
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import RegexpTokenizer

f = codecs.open('/Usuarios/Aealfarop/Documents/Harvard/Crime Classifier/stopwords.txt', 'r', encoding='utf8')
st = f.read()
f.close()
stopwords = set(st.split())
delitos = pn.read_excel('/Usuarios/Aealfarop/Documents/Harvard/Crime Classifier/Crimes.xlsx')
delitos['MODUS OPERANDI'] = delitos['MODUS OPERANDI'].map(lambda x: x.lower())
ddict = list(map(lambda x: x[0], Counter(delitos.DELITO).most_common(9)))
d2i = dict(zip(ddict, range(len(ddict))))
i2d = dict(zip(range(len(ddict)), ddict)) 
#print(d2i)
nwords = 500
stem = True

stemmer = SnowballStemmer("spanish")
tokenizer = RegexpTokenizer('[a-z]+')
def text2tokens(st):
    l = []
    for t in tokenizer.tokenize(st):
        if not t in stopwords:
            l.append(stemmer.stem(t) if stem else t)
    return l

allwords = [t for k in delitos['MODUS OPERANDI'].map(text2tokens).tolist() for t in k]


ct = Counter(allwords)
wdict = list(map(lambda x: x[0], ct.most_common(nwords)))

w2i = dict(zip(wdict, range(len(wdict))))
i2w = dict(zip(range(len(wdict)), wdict))


def text2vec(st):
    vec = np.zeros((nwords,))
    tokens = text2tokens(st)
    idxs = list(map(w2i.get, filter(lambda x: x in w2i, tokens)))
    vec[idxs] = 1
    return vec

#print(w2i)

delitos = delitos[delitos.DELITO.map(lambda x: x in d2i)]

train = delitos[:3250]
test = delitos[3250:]

X_train = np.array(train['MODUS OPERANDI'].map(text2vec).tolist()) 
Y_train = np.array(train['DELITO'].map(d2i.get)) 

bn = BernoulliNB() 
bn.fit(X_train, Y_train) 
pred = bn.predict(X_train) 

X_test = np.array(test['MODUS OPERANDI'].map(text2vec).tolist())
Y_test = np.array(test['DELITO'].map(d2i.get))
test_pred = bn.predict(X_test) 

accuracy = len(list(filter(lambda x: x[0] == x[1], zip(test_pred, Y_test)))) / len(Y_test)
print (accuracy) # result = 0.822742474916388