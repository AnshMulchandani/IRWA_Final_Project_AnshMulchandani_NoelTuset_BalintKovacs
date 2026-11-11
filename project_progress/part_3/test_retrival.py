import part_2.inverted_index as tf_idf
from collections import defaultdict, Counter
from array import array
import math
import numpy as np
import ranking_bm25 as bm25

def compare_retrival(query, index, idf, tf, document_index):
    ranked_docs_bm25 = bm25.search_bm25(query, index, idf, tf, document_index)
    ranked_docs_tf_idf = tf_idf.search_tf_idf(query, index, idf, tf, document_index)
    return