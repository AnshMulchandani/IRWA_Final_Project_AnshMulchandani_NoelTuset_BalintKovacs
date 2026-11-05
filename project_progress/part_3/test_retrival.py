from part_1.preprocess import preprocess_doc, preprocess_line
import part_2.inverted_index as tf_idf
from collections import defaultdict, Counter
from array import array
import math
import numpy as np

def compare_retrival(query, index, idf, tf, document_index):
    ranked_docs_bm25 = search_bm25(query, index, idf, tf, document_index)
    ranked_docs_tf_idf = tf_idf.search_tf_idf(query, index, idf, tf, document_index)
    return


#TO MODIFY for bm25
def search_bm25(query, index, idf, tf, document_index):
    """
    Retrieves matching documents for query terms and ranks them by TF-IDF score.
    """
    cleaned_query = preprocess_line(query)
    if not cleaned_query:
        return []
        
    query_terms = cleaned_query.split()

    #conjunctive search (AND)
    docs = None
    for term in query_terms:
        if term not in index:
            return []
        term_docs = {posting[0] for posting in index[term]}
        docs = term_docs if docs is None else docs.intersection(term_docs)
        if not docs:
            return []
    docs = list(docs)

    #TF-IDF Cosine Similarity Ranking
    ranked_docs = tf_idf.rank_documents_tf_idf(query_terms, docs, index, idf, tf, document_index)
    return ranked_docs