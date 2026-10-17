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

#we download the stowords at the start
nltk.download('stopwords')

#gets the data from the json file
def get_json_data(doc_path):
    
    with open(doc_path, 'r', encoding='utf-8') as fp:
        data = json.load()
    return data

#should we pass data or line by line
def build_terms(data):
    #we initialize the stemmer and stop_word
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    #not finished yet