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

#fields saved in the product record
OUTPUT_FIELDS = ['pid', 'title', 'description', 'brand', 'category', 'sub_category',
                 'product_details', 'seller', 'out_of_stock', 'selling_price',
                 'discount', 'actual_price', 'average_rating', 'url']

#fields to by cleaned
TEXTUAL_FIELDS = ['title', 'description', 'brand', 'sub_category']

#we download the stowords at the start
nltk.download('stopwords')
#we initialize the stemmer and stop_word
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

#gets the data from the json file
def get_json_data(doc_path):
    
    with open(doc_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


#we perform datacleaning for title and description
def preprocess_line(line):
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
    #join the words back to a string
    line = ' '.join(line)
    return line

def clean_product_details(details):
    """
    Turn product_details, a list of one-key dicts, into one dict.

    Example: [{'Fabric': 'Cotton'}, {'Pattern': 'Solid'}] -> {'Fabric': 'Cotton', 'Pattern': 'Solid'}
    27 products repeat a key; their values are joined with ' ' so nothing is lost.
    """
    if isinstance(details, dict):
        return dict(details)  #already flattened
    flat = {}
    for entry in details or []:
        for key, value in entry.items():
            key = preprocess_line(key)
            value = preprocess_line(value)
            flat[key] = value if key not in flat else flat[key] + ' ' + value
    return flat

def parse_price(raw):
    """'2,999' -> 2999.0. Empty or unreadable -> None."""
    try:
        return float(str(raw).replace(',', '').strip())
    except ValueError:
        return None


def parse_discount(raw):
    """'69% off' -> 69.0. Empty or unreadable -> None."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)  #already parsed
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', str(raw or ''))
    return float(match.group(1)) if match else None


def parse_rating(raw):
    """'3.9' -> 3.9. Empty or unreadable -> None."""
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def parse_numeric_fields(product):
    """Parse the five non-text fields of point 4 into numbers and a boolean."""
    return {
        'out_of_stock': bool(product.get('out_of_stock', False)),
        'selling_price': parse_price(product.get('selling_price')),
        'actual_price': parse_price(product.get('actual_price')),
        'discount': parse_discount(product.get('discount')),
        'average_rating': parse_rating(product.get('average_rating')),
    }

def build_document_record(product):
    """
    The record stored for one product (point 2): the 14 output fields, with
    product_details as a dict and the point 4 fields parsed into numbers.
    """
    record = {}
    for field in OUTPUT_FIELDS:
        product_data = product.get(field) or ''

        if field in TEXTUAL_FIELDS:
            product_data = preprocess_line(product_data)

        record[field] = product_data

    record['product_details'] = clean_product_details(product.get('product_details'))
    record.update(parse_numeric_fields(product))
    return record



#Creates a dictionary mapping each pid to its full cleaned record (Points 2 & 4) and the tokenized text fields ready for indexing (Points 1 & 3).
def preprocess_doc(data):
    document_index = {}

    for product in data:

        prod_pid = product['pid']
        #Points 2 & 4: Build base dictionary with the 14 output fields,
        #flattened product_details, and parsed numeric/boolean values
        record = build_document_record(product)

        document_index[prod_pid] = record
        
    return document_index