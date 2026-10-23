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

from fields import (
    build_document_record,
    identifier_detail_keys,
    numeric_terms,
    searchable_text,
)

#we download the stowords at the start
nltk.download('stopwords')
#we initialize the stemmer and stop_word
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

#gets the data from the json file
def get_json_data(doc_path):
    
    with open(doc_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)


#we perform datacleaning for title and description
def build_terms(line):
    #in case we don't get anything
    if not isinstance(line, str):
        return []
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

#Creates a dictionary mapping each pid to its full cleaned record (Points 2 & 4) and the tokenized text fields ready for indexing (Points 1 & 3).
def preprocess_doc(data, numeric_as_terms=False):
    document_index = {}
    skip_keys = identifier_detail_keys(data)

    for product in data:

        prod_pid = product['pid']
        # Points 2 & 4: Build base dictionary with the 14 output fields,
        # flattened product_details, and parsed numeric/boolean values
        record = build_document_record(product)
        
        # Point 1: Tokenized title and description
        record['title_tokens'] = build_terms(product.get('title', ''))
        record['description_tokens'] = build_terms(product.get('description', ''))
        
        # Point 3: Tokenized hybrid searchable text (for Part 2 indexing)
        record['searchable_tokens'] = build_terms(searchable_text(product, skip_keys))
        
        # Point 4 option (off by default): append numeric band terms
        if numeric_as_terms:
            record['searchable_tokens'] = record['searchable_tokens'] + numeric_terms(record)
            
        document_index[prod_pid] = record
        
    return document_index