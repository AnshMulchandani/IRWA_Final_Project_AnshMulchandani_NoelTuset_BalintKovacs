import nltk
from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import json #to handle data in json format
import re #to remove punctuation

#we download the stowords at the start
nltk.download('stopwords')

#we perform preprocessing for title and description
def build_terms(line):
    #we initialize the stemmer and stop_word
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    #we lower the strings
    line = line.lower()
    #puntiation like - or ' is removed without a space then t-shirt becomes tshirt or men's becomes mens
    line = re.sub(r"['\-]", "", line)
    #removes punctuations, any alfanumeric matches r'[^\w\s]', else it substitutes puntiation with blank space
    line = re.sub(r'[^\w\s]', ' ', line)
    line = line.split()
    #remove stop words
    line = [x for x in line if x not in stop_words ]
    #perform steaming
    line=[stemmer.stem(x) for x in line]
    return line

#we perform the preprocessing in the entire document and we return the document with the title and description preprocessed
def preprocess_doc(data):
    for product in data:
        product.get('title', '') = build_terms(product.get('title', ''))
        product.get('description', '') = build_terms(product.get('description', ''))
    return data

