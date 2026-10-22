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

<<<<<<< Updated upstream
#gets the data from the json file
def get_json_data(doc_path):
    
    with open(doc_path, 'r', encoding='utf-8') as fp:
        data = json.load()
=======
#'t shirt' (with a space, 1,584 products) would lose 't' as a stop word and become 'shirt',
#the same term as a plain shirt; this maps every spelling to 'tshirt' first
NORMALISE_TSHIRT = True

#we perform preprocessing for title and description
def build_terms(line):
    #we lower the strings
    line = line.lower()
    if NORMALISE_TSHIRT:
        line = re.sub(r'\bt[\s-]?shirts?\b', 'tshirt', line)
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
def preprocess_doc(data, numeric_as_terms=False):
    #product_details keys that hold codes (like Style Code) stay out of processed_terms
    skip_keys = identifier_detail_keys(data)
    for product in data:
        # get the title and index from the 
        title_text = product.get('title', '')
        desc_text = product.get('description', '')
        
        # Store the preprocessed elements in new campus
        product['processed_title'] = build_terms(title_text)
        product['processed_description'] = build_terms(desc_text)
        #point 3: the words of every searchable field (title, description, brand, sub_category
        #and the product_details values); this is what Part 2 indexes
        product['processed_terms'] = build_terms(searchable_text(product, skip_keys))
        #points 2 and 4: product_details as one dict, numeric fields parsed (None when missing)
        product['product_details'] = flatten_product_details(product.get('product_details'))
        product.update(parse_numeric_fields(product))
        #point 4 option (off by default): numeric fields as band terms like price_500_999
        if numeric_as_terms:
            product['processed_terms'] = product['processed_terms'] + numeric_terms(product)
        
>>>>>>> Stashed changes
    return data

#we perform datacleaning for title and description
def build_terms(line):
    #we initialize the stemmer and stop_word
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    #we lower the strings
    line = line.lower()
    #removes punctuations, any alfanumeric matches r'[^\w\s]', else it substitutes puntiation with blank space
    line = re.sub(r'[^\w\s]', '', line)
    line = line.split()
    #remove stop words
    line = [x for x in line if x not in stop_words ]
    #perform steaming
    line=[stemmer.stem(x) for x in line]
    return line

def create_index(data):
    index = defaultdict(list)
    title_index = {}
    for product in data:
        #get the document id
        prod_pid = product['pid']
        #passes the title + description, in case there isn't title or description we don't pass anything
        terms = build_terms(product.get('title', '') + " " + product.get('description', ''))
        title = product.get('title', '')
        title_index[prod_pid] = title
        current_page_index = {}
        for position, term in enumerate(terms): #term is title + description
            try:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list
                     current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[prod_pid, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(current_page_index[term_page])

    return index, title_index


