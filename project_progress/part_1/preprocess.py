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
#we initialize the stemmer and stop_word
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

#we perform preprocessing for title and description
def build_terms(line):
    #we lower the strings
    line = line.lower()
    #puntiation like - or ' is removed without a space then t-shirt becomes tshirt or men's becomes mens
    line = re.sub(r"['\-]", "", line)
    #removes punctuations, any alfanumeric matches r'[^\w\s]', else it substitutes puntiation with blank space
    line = re.sub(r'[^\w\s]', ' ', line)
    line = line.split()
    #remove stop words
    line = [x for x in line if x not in STOP_WORDS ]
    #perform steaming
    line=[STEMMER.stem(x) for x in line]
    return line

#we perform the preprocessing in the entire document and we return the document with the title and description preprocessed
def preprocess_doc(data):
    for product in data:
        # get the title and index from the 
        title_text = product.get('title', '')
        desc_text = product.get('description', '')
        
        # Store the preprocessed elements in new campus
        product['processed_title'] = build_terms(title_text)
        product['processed_description'] = build_terms(desc_text)
        
    return data

