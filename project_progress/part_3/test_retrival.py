from part_1.preprocess import preprocess_line
import part_2.inverted_index as tf_idf
from collections import defaultdict, Counter
from array import array
import math
import numpy as np

# Cache for document lengths so we don't compute it multiple times.
DOC_LENGTHS_CACHE = {}
AVGDL_CACHE = 0.0

def get_doc_lengths(index, num_documents):
    """
    Computes document lengths directly from the inverted index.
    A document's length is the total number of term positions it contains.
    """
    global AVGDL_CACHE
    if not DOC_LENGTHS_CACHE:
        # Reconstruct length from index instead of re-parsing text
        for term, postings in index.items():
            for doc_id, positions in postings:
                # The length of the positions array is how many times the term appeared
                DOC_LENGTHS_CACHE[doc_id] = DOC_LENGTHS_CACHE.get(doc_id, 0) + len(positions)
                
        total_length = sum(DOC_LENGTHS_CACHE.values())
        AVGDL_CACHE = total_length / num_documents if num_documents > 0 else 1.0
        
    return DOC_LENGTHS_CACHE, AVGDL_CACHE

def rank_documents_bm25(query_terms, docs, index, document_index, k1=1, b=0.75):
    """
    Ranks matching documents using BM25 from theory
    """
    N = len(document_index)
    doc_lengths, avgdl = get_doc_lengths(index, N)
    
    doc_scores = defaultdict(float)
    
    for term in query_terms:
        if term not in index:
            continue
            
        # IDF is: log_2(N / df_t)
        df = len(index[term])
        idf_bm25 = math.log2(N / df)
        
        # Iterate through matched documents to calculate term scores
        for doc_id, postings in index[term]:
            if doc_id in docs:
                #we get the term frequency from the number of times it appear in a document
                tf_td = len(postings)
                #doc len is stored in posicion doc_id if not found it has value average lenght
                doc_len = doc_lengths.get(doc_id, avgdl)
                
                # Document length normalization factor K 
                # K = k1 * ((1 - b) + b * (L_d / L_ave))
                K = k1 * ((1 - b) + b * (doc_len / avgdl))
                
                # Term frequency component
                # TF(t,d) = ((k1 + 1) * tf_td) / (K + tf_td)
                tf_component = ((k1 + 1) * tf_td) / (K + tf_td)
                
                # Final score
                # Contribution = IDF(t) * TF(t,d)
                doc_scores[doc_id] += idf_bm25 * tf_component
                
    # Sort documents by descending RSV_d score and returns a list with [doc_id, score]
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    #returns a list of doc_id sorted by descending RSV_d
    return [doc_id for doc_id, score in sorted_docs]


def search_bm25(query, index, document_index):
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
    ranked_docs = rank_documents_bm25(query_terms, docs, index, document_index)
    return ranked_docs

def compare_retrival(query, index, idf, tf, document_index):
    ranked_docs_bm25 = search_bm25(query, index, idf, tf, document_index)
    ranked_docs_tf_idf = tf_idf.search_tf_idf(query, index, idf, tf, document_index)
    return