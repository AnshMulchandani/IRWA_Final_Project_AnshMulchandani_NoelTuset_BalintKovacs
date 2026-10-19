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

#gets the data from the json file
def get_json_data(doc_path):
    
    with open(doc_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
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
    document_index = {}
    for product in data:
        #get the document id
        prod_pid = product['pid']
        #passes the title + description, in case there isn't title or description we don't pass anything
        terms = build_terms(product.get('title', '') + " " + product.get('description', ''))
        title = product.get('title', '')
        document_index[prod_pid] = { prod_pid,
            title,
            product.get('description', ''),
            product.get('brand', ''),
            product.get('category', ''),
            product.get('sub_category', ''),
            product.get('product_details', []),
            product.get('seller', ''),
            product.get('out_of_stock', False), #by default we asume we have stock of a product
            product.get('selling_price', ''),
            product.get('discount', ''),
            product.get('actual_price', ''),
            product.get('average_rating', ''),
            product.get('url', '')
            }
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

    return index, document_index


