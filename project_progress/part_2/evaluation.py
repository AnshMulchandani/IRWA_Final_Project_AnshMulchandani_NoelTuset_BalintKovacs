import os
import pandas as pd
from part_2.inverted_index import create_index_tfidf, search_tf_idf 
from part_1.preprocess import get_json_data
# Get the absolute path to this document, then go one folder up to the parent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Computes Precision@K.
    """
    if not retrieved_docs:
        return 0.0
        
    # Isolate the top K retrieved documents
    top_k_retrieved = retrieved_docs[:k]
    
    # Count how many of the top K are in the relevant_docs ground truth
    relevant_set = set(relevant_docs)
    hits = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    return hits / k



def main():
    dataset_file = os.path.join(ROOT_DIR, "data", "fashion_products_dataset.json")
    validation_file = os.path.join(ROOT_DIR, "data", "validation_labels.csv")
    K_VALUES = [5, 10]
    
    print(f"Loading dataset from {dataset_file}...")
    dataset = get_json_data(dataset_file)
        
    print("Building TF-IDF Inverted Index...")
    index, tf, df, idf, document_index = create_index_tfidf(dataset)
    print(f"Index built successfully for {len(document_index)} documents.\n")

    print(f"Loading validation labels from {validation_file}...")
    try:
        val_df = pd.read_csv(validation_file)
    except FileNotFoundError:
        print(f"Error: Could not find {validation_file}.")
        return

    # 1. Map query_ids to actual text queries. 
    # Since your CSV only has 'query_id', you must supply the text to search for.
    # (If you have a separate queries.csv file, you can load it into a dict here instead)
    queries_map = {
        1: "women sweatshirt", # Replace with the actual search text for query_id 1
        2: "men track pants",  # Replace with the actual search text for query_id 2
    }

    cumulative_precisions = {k: 0.0 for k in K_VALUES}
    num_queries = 0

    print("Evaluating queries...")
    
    # 2. Group the data by 'query_id' to evaluate one search at a time
    for query_id, group in val_df.groupby('query_id'):
        if query_id not in queries_map:
            print(f"Warning: No text provided in queries_map for query_id {query_id}. Skipping.")
            continue
            
        query_text = queries_map[query_id]
        
        # 3. Filter for relevant documents (labels == 1) and extract their PIDs into a list
        relevant_docs = group[group['labels'] == 1]['pid'].tolist()
        
        # Skip if a query has no relevant documents in the dataset
        if not relevant_docs:
            continue
            
        # Run the search
        retrieved_docs = search_tf_idf(query_text, index, idf, tf, document_index)
        
        # Calculate Precision for each K
        for k in K_VALUES:
            p_at_k = calculate_precision_at_k(retrieved_docs, relevant_docs, k)
            cumulative_precisions[k] += p_at_k
            
        num_queries += 1

    # Print results
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"Total Queries Evaluated: {num_queries}")
    
    if num_queries > 0:
        for k in K_VALUES:
            avg_precision = cumulative_precisions[k] / num_queries
            print(f"Average Precision@{k}: {avg_precision:.4f}")

if __name__ == "__main__":
    main()