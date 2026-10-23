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

from fields import flatten_product_details, identifier_detail_keys, numeric_terms, parse_numeric_fields, searchable_text

#we download the stowords at the start
nltk.download('stopwords')
#we initialize the stemmer and stop_word
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

#gets the data from the json file
def get_json_data(doc_path):
    
    with open(doc_path, 'r', encoding='utf-8') as fp:
        data = json.load()


#we perform datacleaning for title and description
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

# creates a dictonary to store relevant data for queries and outputs
def preprocess_doc(data, numeric_as_terms=False):
    document_index = {}
    skip_keys = identifier_detail_keys(data)
    for product in data:
        #get the document id
        prod_pid = product['pid']
        document_index[prod_pid] = { prod_pid,
            build_terms(product.get('title', '')),
            build_terms(product.get('description', '')),
            #point 3: the words of every searchable field (title, description, brand, sub_category
            #and the product_details values); this is what Part 2 indexes
            build_terms(searchable_text(product, skip_keys)),
            #points 2 and 4: product_details as one dict, numeric fields parsed (None when missing)
            flatten_product_details(product.get('product_details')),
            product.update(parse_numeric_fields(product)),
        }
        #point 4 option (off by default): numeric fields as band terms like price_500_999
        if numeric_as_terms:
            product['pid'][4] = product['pid'][4] + numeric_terms(product)
    return document_index