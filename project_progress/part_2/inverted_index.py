from part_1.preprocess import preprocess_doc, preprocess_line
from collections import defaultdict, Counter
from array import array
import math
import numpy as np



def create_index_tfidf(data):
    """
    Build inverted index and calculate TF, DF, IDF, and document index
    """
    index = defaultdict(list)
    tf = defaultdict(list)    #tf[term] stores list of normalized TFs parallel to index[term]
    df = defaultdict(int)     #df[term] stores document frequency
    idf = defaultdict(float)  #idf[term] stores inverse document frequency
    
    #preprocess all documents into a dict of prod_id -> record
    document_index = preprocess_doc(data)
    num_documents = len(document_index)

    #for each doc
    for prod_id, record in document_index.items():
        #convert all text fields into a list of terms
        product_text = ' '.join(value for key, value in (record.get('product_details', {})).items())
        full_text = f"{record.get('title', '')} {record.get('description', '')} {record.get('brand', '')} {record.get('sub_category', '')} {product_text}"
        terms = full_text.split()
        
        if not terms:
            continue

        #build local page index: term -> [prod_id, array of positions]
        current_page_index = {}
        for position, term in enumerate(terms):
            try:
                current_page_index[term][1].append(position)
            except KeyError:
                current_page_index[term] = [prod_id, array('I', [position])]

        #compute document L2 norm denominator
        norm = 0
        for term, posting in current_page_index.items():
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        #calculate normalized TF, update DF and merge local page index into global index
        for term, posting in current_page_index.items():
            tf[term].append(np.round(len(posting[1]) / norm, 4) if norm > 0 else 0.0)
            df[term] += 1
            index[term].append(posting)

    #compute IDF for each term 
    for term in df:
        idf[term] = np.round(np.log(float(num_documents / df[term])), 4)

    return index, tf, df, idf, document_index


def rank_documents(terms, docs, index, idf, tf, document_index):
    """
    Perform ranking of candidate documents matching a query based on TF-IDF cosine similarity
    """
    #initialize document vectors (with 0 for each query term) and query vector
    doc_vectors = defaultdict(lambda: [0.0] * len(terms))
    query_vector = [0.0] * len(terms)

    #compute L2 norm for the query terms
    query_terms_count = Counter(terms)
    query_norm = np.linalg.norm(list(query_terms_count.values()))

    #build Query Vector & Document Vectors for matching terms
    for termIndex, term in enumerate(terms):
        if term not in index:
            continue

        #compute normalized TF-IDF score for query vector entry
        #query TF = count / query_norm, multiplied by IDF
        query_tf = (query_terms_count[term] / query_norm) if query_norm > 0 else 0.0
        query_vector[termIndex] = query_tf * idf[term]

        # Generate doc_vectors for matching documents
        for doc_index, (doc, postings) in enumerate(index[term]):
            if doc in docs:
                #doc_vectors[doc][termIndex] = TF * IDF
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    #calculate score using dot product (cosine similarity of normalized vectors)
    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    
    result_docs = [x[1] for x in doc_scores]
    return result_docs


def search_tf_idf(query, index, idf, tf, document_index):
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
    ranked_docs = rank_documents(query_terms, docs, index, idf, tf, document_index)
    return ranked_docs