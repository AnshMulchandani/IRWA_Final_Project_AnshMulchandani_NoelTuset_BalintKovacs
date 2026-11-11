import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_progress_dir = os.path.abspath(os.path.join(script_dir, ".."))
root_dir = os.path.abspath(os.path.join(project_progress_dir, ".."))

for path in [project_progress_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from part_1.preprocess import preprocess_doc, preprocess_line
from collections import defaultdict, Counter
from array import array
import math
import numpy as np


def calculate_custom_multiplier(record, a=1.0, b=2.0):
    """
    Calculates the value multiplier combining discount logic and stock penalty
    """
    try:
        #get prices to calculate discount percentage and absolute cash saved
        actual_price = record.get('actual_price', 0.0)
        selling_price = record.get('selling_price', 0.0)
        discount = record.get('discount', 0.0)
            
        money_saved = actual_price - selling_price
        
        value_score = (discount * a) + (np.log(money_saved + 1) * b)
        
        multiplier = 1.0 + (value_score / 100.0)
        
    except (ValueError, TypeError):
        multiplier = 1.0

    #apply out of stock penalty
    is_out_of_stock = record.get('out_of_stock', False) == True
    stock_penalty = 0.8 if is_out_of_stock else 1.0

    return multiplier * stock_penalty

def create_index_tfidf(data, field_weights=None):
    """
    Build inverted index and calculate TF, DF, IDF, and document index with custom business weights.
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
        #compute and store the custom multiplier for this document
        record['value_multiplier'] = calculate_custom_multiplier(record)

        field_texts = {
            'title': record.get('title', ''),
            'brand': record.get('brand', ''),
            'sub_category': record.get('sub_category', ''),
            'description': record.get('description', ''),
            'product_details': ' '.join(value for key, value in (record.get('product_details', {})).items())
        }

        
        #build local page index: term -> [prod_id, array of positions]
        current_page_index = {}


        for field_name, text in field_texts.items():
            weight = field_weights.get(field_name, 1.0)
            for position, term in enumerate(text.split()):
                if term not in current_page_index:
                    current_page_index[term] = [prod_id, array('I', [position]), array('f', [weight])]
                else:
                    current_page_index[term][1].append(position)
                    current_page_index[term][2].append(weight)

        if not current_page_index:
            continue

        #compute document L2 norm denominator with weighted counts
        norm = 0
        for term, posting in current_page_index.items():
            weighted_count = sum(posting[2])
            norm += weighted_count ** 2
        norm = math.sqrt(norm)

        #calculate normalized TF, update DF and merge local page index into global index
        for term, posting in current_page_index.items():
            weighted_count = sum(posting[2])
            tf[term].append(np.round(weighted_count / norm, 4) if norm > 0 else 0.0)
            df[term] += 1
            index[term].append(posting)

    #compute IDF for each term 
    for term in df:
        idf[term] = np.round(np.log2(float(num_documents / df[term])), 4)

    return index, tf, df, idf, document_index




def rank_documents_tf_idf(terms, docs, index, idf, tf, document_index):
    """
    Perform ranking of candidate documents matching a query based on TF-IDF cosine similarity,
    factoring in the precalculated business logic multiplier.
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
        for doc_index, (doc, _, _) in enumerate(index[term]):
            if doc in docs:
                #doc_vectors[doc][termIndex] = TF * IDF
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]

    #calculate score using dot product (cosine similarity of normalized vectors)
    #multiplied by the custom precalculated multiplier
    doc_scores = [
        [np.dot(curDocVec, query_vector) * document_index[doc].get('value_multiplier', 1.0), doc] 
        for doc, curDocVec in doc_vectors.items()
    ]
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

    #TF-IDF Cosine Similarity Ranking with Custom Multiplier
    ranked_docs = rank_documents_tf_idf(query_terms, docs, index, idf, tf, document_index)
    return ranked_docs





if __name__ == "__main__":
    import json
    import sys

    #setup paths relative to script execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_progress_dir = os.path.abspath(os.path.join(script_dir, ".."))
    root_dir = os.path.abspath(os.path.join(project_progress_dir, ".."))

    dataset_path = os.path.join(root_dir, "data", "fashion_products_dataset.json")

    print(f"--- Running Direct Execution Test for Custom Ranking ---")
    print(f"Loading dataset from: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    print(f"Loaded {len(sample_data)} products successfully.")

    #define custom field weights for ranking
    FIELD_WEIGHTS = {
        'title': 4.0,
        'brand': 3.0,
        'sub_category': 2.0,
        'description': 1.0,
        'product_details': 1.0
    }
    

    #build inverted index and calculate TF, DF, IDF, and document index with custom weights
    index, tf, df, idf, document_index = create_index_tfidf(sample_data, FIELD_WEIGHTS)
    print(f"Inverted index constructed ({len(index)} terms).\n")

    test_queries = [
        "women tshirt round neck cotton",
        "ecko unl printed sweatshirt",
        "men solid blue track pants",
        "satin necktie set pocket square cufflinks",
        "checkered full sleeve party blazer"
    ]

    print("================ EVALUATING TEST QUERIES ================")
    for q in test_queries:
        results = search_tf_idf(q, index, idf, tf, document_index)
        print(f"\nQuery: '{q}' (Total Matches: {len(results)})")
        
        for rank, pid in enumerate(results[:5], start=1):
            doc = document_index[pid]
            title = doc.get('title', 'No Title')
            multiplier = doc.get('value_multiplier', 1.0)
            actual = doc.get('actual_price', 0.0)
            selling = doc.get('selling_price', 0.0)
            discount = doc.get('discount', 0.0)
            stock_status = "OUT" if (str(doc.get('out_of_stock', '')).lower() == 'true' or str(doc.get('stock_status', '')).lower() == 'out_of_stock') else "IN"
            
            print(f"  {rank}. [{pid}] {title}")
            print(f"      └─ Multiplier: {multiplier:.4f} | Stock: {stock_status} | Actual: {actual} | Selling: {selling} | Discount: {discount}%")