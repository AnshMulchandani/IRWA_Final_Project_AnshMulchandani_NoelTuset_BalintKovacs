import os
import pandas as pd
from part_2.inverted_index import create_index_tfidf, search_tf_idf 
from part_1.preprocess import get_json_data, get_csv_data
import math
# Get the absolute path to this document, then go one folder up to the parent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Computes Precision@K, this is done by retriving how many documents in top K are relevanr
    and then it computes relevant_documents_in_top_k/K.
    """
    if not retrieved_docs:
        return 0.0
        
    # Isolate the top K retrieved documents
    top_k_retrieved = retrieved_docs[:k]
    
    # Count how many of the top K are in the relevant_docs ground truth
    relevant_set = set(relevant_docs)
    hits = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    return hits / k

def calculate_recall_at_k(retrieved_docs, relevant_docs, k):
    """
        Computes Recall@K, this is done by retriving how many documents in top K are relevanr
        and then it computes relevant_documents_in_top_k/total_relevant_docs.
        """
    if not retrieved_docs:
            return 0.0
            
    # Isolate the top K retrieved documents
    top_k_retrieved = retrieved_docs[:k]
        
    # Count how many of the top K are in the relevant_docs ground truth
    relevant_set = set(relevant_docs)
    hits = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    total_relevant = len(relevant_docs)
        
    return hits / total_relevant

def calculate_f1_score_at_k(p_at_k, r_at_k):
    """
    Computes F1-Score@K using the previously calculated Precision@K and Recall@K.
    """
    if p_at_k + r_at_k == 0:
        return 0.0
        
    return 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k)

def calculate_average_precision(retrieved_docs, relevant_docs):
    """
    Computes the Average Precision (AP) for a single query, used later on for MAP computation.
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0
        
    relevant_set = set(relevant_docs)
    hits = 0
    sum_precisions = 0.0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            hits += 1
            # Precision at this specific hit position
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
            
    # Divide by the total number of relevant documents for this query
    return sum_precisions / len(relevant_docs)

def calculate_reciprocal_rank(retrieved_docs, relevant_docs):
    """
    Computes the Reciprocal Rank (RR) for a single query.
    MRR is obtained by averaging this value across all queries.
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0
        
    relevant_set = set(relevant_docs)
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            # i is 0-indexed, so the actual rank is i + 1
            return 1.0 / (i + 1)
            
    return 0.0

def calculate_ndcg_at_k(retrieved_docs, relevant_docs, k):
    """
    Computes NDCG@K. We assume binary relevance (1 if relevant, 0 if not).
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0
        
    relevant_set = set(relevant_docs)
    
    # 1. Calculate DCG@K
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_set:
            # Rank is i + 1. log2(rank + 1) becomes log2(i + 2)
            dcg += 1.0 / math.log2(i + 2)
            
    # 2. Calculate Ideal DCG@K (IDCG)
    # The ideal ranking has the relevant documents first and then the non relevant
    idcg = 0.0
    ideal_hits = min(len(relevant_docs), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def main():
    dataset_file = os.path.join(ROOT_DIR, "data", "fashion_products_dataset.json")
    validation_file = os.path.join(ROOT_DIR, "data", "validation_labels.csv")
    K = 10 #K value, it can be tunned
    
    print(f"Loading dataset from {dataset_file}...")
    dataset = get_json_data(dataset_file)
        
    print("Building TF-IDF Inverted Index...")
    index, tf, df, idf, document_index = create_index_tfidf(dataset)
    print(f"Index built successfully for {len(document_index)} documents.\n")

    print(f"Loading validation labels from {validation_file}...")
    val_df = get_csv_data(validation_file)

    # 1. Map query_ids to actual text queries. 
    # Since your CSV only has 'query_id', introduce the text to search for.
    queries_map = {
        1: "women full sleeve sweatshirt cotton", 
        2: "men slim jeans blue",  
    }

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_f1 = 0.0
    cumulative_ndcg = 0.0
    
    cumulative_ap = 0.0
    cumulative_rr = 0.0
    num_queries = 0

    print("Evaluating queries...")
    
    #Group the data by 'query_id' to evaluate one search at a time
    for query_id, group in val_df.groupby('query_id'):
        if query_id not in queries_map:
            print(f"Warning: No text provided in queries_map for query_id {query_id}. Skipping.")
            continue
            
        query_text = queries_map[query_id]
        
        # Filter for relevant documents (labels == 1) and extract their PIDs into a list
        relevant_docs = group[group['labels'] == 1]['pid'].tolist()
        
        # Skip if a query has no relevant documents in the dataset
        if not relevant_docs:
            continue
        # Run the search
        retrieved_docs = search_tf_idf(query_text, index, idf, tf, document_index)
        print(f"\n--- Results for Query: '{query_text}' (ID: {query_id}) ---")
        # Calculate overall query metrics
        ap = calculate_average_precision(retrieved_docs, relevant_docs)
        rr = calculate_reciprocal_rank(retrieved_docs, relevant_docs)
        cumulative_ap += ap
        cumulative_rr += rr
        # Calculate Precision for each query and K
        p_at_k = calculate_precision_at_k(retrieved_docs, relevant_docs, K)
        r_at_k = calculate_recall_at_k(retrieved_docs, relevant_docs, K)
        f1_at_k = calculate_f1_score_at_k(p_at_k, r_at_k)
        ndcg_at_k = calculate_ndcg_at_k(retrieved_docs, relevant_docs, K)
        cumulative_precision += p_at_k
        cumulative_recall += r_at_k
        cumulative_f1 += f1_at_k
        cumulative_ndcg += ndcg_at_k
        print(f"  Precision@{K}: {p_at_k:.3f}")
        print(f"  Recall@{K}: {r_at_k:.3f}")
        print(f"  NDCG@{K}:      {ndcg_at_k:.3f}")
        num_queries += 1

    # Print results
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Total Queries Evaluated: {num_queries}")
    if num_queries > 0:
        print(f"Average Precision@{K}: {cumulative_precision / num_queries:.3f}")
        print(f"Average Recall@{K}:    {cumulative_recall / num_queries:.3f}")
        print(f"Average F1-Score@{K}:  {cumulative_f1 / num_queries:.3f}")
        print(f"Mean NDCG@{K}:         {cumulative_ndcg / num_queries:.3f}")
        print(f"Mean Average Precision (MAP): {cumulative_ap / num_queries:.3f}")
        print(f"Mean Reciprocal Rank (MRR):   {cumulative_rr / num_queries:.3f}")

if __name__ == "__main__":
    main()