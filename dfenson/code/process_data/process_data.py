import random
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from os import listdir
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer,\
    VectorizerMixin
from pathlib import Path 
#import stop_words
from sklearn.decomposition import PCA, TruncatedSVD
from collections import OrderedDict
import pickle
from sklearn.decomposition.incremental_pca import IncrementalPCA
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
#from stop_words import STOP_WORDS

# Word Feature Matrix processing.
def featurize_exerpts(excerpts):
    vectorizer = CountVectorizer(decode_error='replace',
                                 input='content',
                                 stop_words=STOP_WORDS,
                                 min_df=0,
                                 max_df=1,
                                 tokenizer=Tokenizer())
    X = vectorizer.fit_transform(excerpts)
    words = vectorizer.get_feature_names()
    return X, words, vectorizer

#Word Feature Matrix with stemming
def stem_excerpts(excerpts):
    X, words, vectorizer=featurize_exerpts(excerpts)
    X, words = stem_design_matrix(X, words)
    return X, words, vectorizer

# Demension reduction
def reduce_feature_matrix(X, n_components, reducer=PCA):
    reductionObj = reducer(n_components = n_components)
    X = pca.fit_transform(X)
    return X

#TFIDF processing
def tfidf(X):
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    return X, transformer

# Remove words with numerals
def remove_numerals(X, words):
    merged_columns = OrderedDict()
    for i in range(len(words)):
        if not re.match("\A\d*\Z", words[i]):
            merged_columns[words[i]] = X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()

#Power feature, if document contains numeral
def numeral_feature(X, words):
    numerals = np.zeros(X.shape[0])
    for i in range(len(words)):
        if re.match("[0-9]+", words[i]):
            numerals = np.add(X[:, i], numerals)
    return numerals

# Stemming
def stem_design_matrix(X, words, type='snowball'):
    if type == 'snowball':
        stemmer = SnowballStemmer("english")
    elif type == 'porter':
        stemmer = PorterStemmer()
    else:
        stemmer = LancasterStemmer()
    words = [stemmer.stem(w) for w in words]
    merged_columns = OrderedDict()
    for i in range(len(words)):
        if words[i] not in merged_columns:
            merged_columns[words[i]] = X[:, i]
        else:
            merged_columns[words[i]] += X[:, i]
    return np.array(merged_columns.values()).T, merged_columns.keys()
    

# Assume that X is our numpy array design matrix, header is a list of our features
def write_to_csv(filename, X, header):
    np.savetxt(filename,
               X,
               fmt="%d",
               delimiter=",",
               header=",".join(header),
               comments="")

class Tokenizer:

    PART = {
        'N':'n',
        'V':'v',
        'J':'a',
        'S':'s',
        'R':'r'
    }

    def __call__(self, doc, lemmatize=True):
        words=self.tokenize(doc)
        if lemmatize:
            return self.lemmatize(words)
        else:
            return words

    def tokenize(self, doc):
        return [self.strip(t) for t in word_tokenize(doc) if len(self.strip(t)) >= 2]

    def strip(self, word):
        return re.sub('[\W_]+', '', word)

    def lemmatize(self, words, only_nouns=False):
        wnl = WordNetLemmatizer()
        if only_nouns:
            return wnl.lemmatize(words)
        else:
            words = self.tag_words(words)
            words = [wnl.lemmatize(w, t) for w,t in words]
            return list(set(words))

    # text is a list that contains all text from a single document. Assume no non alpha-numeric text.
    def tag_words(self,words):
        tags = pos_tag(words)
        tags = [(word, self.convert_tags(tag)) for (word, tag) in tags]
        return tags

    def convert_tags(self,tag):
        if tag[0] in self.PART:
            return self.PART[tag[0]]
        else:
            return 'n'

if __name__=="__main__":
    dataPath="/Users/derek/public_comments/Public_Comments_Tool/Common/excerpt_data"
    fullTrainPath="%s/excerpts_training_data.sas7bdat" % dataPath
    fullMatrixPath="%s/word_count_matrix.csv" % dataPath
    df=pd.read_sas(fullTrainPath, encoding="ISO-8859-1")
    excerpts=list(df.Comment_Excerpt)
    X, words, vect=featurize_exerpts(excerpts)
    write_to_csv(fullMatrixPath, X, words)




