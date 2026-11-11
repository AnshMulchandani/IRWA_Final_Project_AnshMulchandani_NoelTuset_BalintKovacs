from part_1.preprocess import get_json_data
import part_2.inverted_index as tf_idf
import part_3.ranking_bm25 as bm25
import part_3.custom_ranking as custom
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

def compare_retrival(query, index, idf, tf, document_index, index_custom, idf_custom, tf_custom, document_index_custom, top_k=10):
    """
    Executes all three search algorithms, prints the top results side by side, 
    and calculates the document overlap among the top K results.
    """
    print(f"\n{'='*70}\nQuery: '{query}'\n{'='*70}")

    # Execute searches using the appropriate index for each algorithm
    ranked_docs_bm25 = bm25.search_bm25(query, index, document_index)
    ranked_docs_tf_idf = tf_idf.search_tf_idf(query, index, idf, tf, document_index)
    ranked_docs_custom = custom.search_tf_idf_custom(query, index_custom, idf_custom, tf_custom, document_index_custom)

    # Grab the top K results
    top_bm25 = ranked_docs_bm25[:top_k]
    top_tf_idf = ranked_docs_tf_idf[:top_k]
    top_custom = ranked_docs_custom[:top_k]

    # Print TF-IDF 
    print(f"\n--- Top {top_k} Standard TF-IDF Results ---")
    #shows results in order starting from 1 to 10
    for rank, doc_id in enumerate(top_tf_idf, 1):
        title = document_index[doc_id].get('title', 'No Title Found')
        print(f" {rank:2d}. [ID: {doc_id}] {title}")
        
    # Print BM25
    print(f"\n--- Top {top_k} BM25 Results ---")
    for rank, doc_id in enumerate(top_bm25, 1):
        title = document_index[doc_id].get('title', 'No Title Found')
        print(f" {rank:2d}. [ID: {doc_id}] {title}")

    # Print Custom
    print(f"\n--- Top {top_k} Custom Ranking Results ---")
    for rank, doc_id in enumerate(top_custom, 1):
        # Fetching from document_index_custom to show the custom multiplier
        doc = document_index_custom[doc_id]
        title = doc.get('title', 'No Title Found')
        multiplier = doc.get('value_multiplier', 1.0)
        print(f" {rank:2d}. [ID: {doc_id}] {title} (Multiplier: {multiplier:.2f})")

    # Calculate Jaccard Overlap
    # Calculates how many identical documents appear in the top K across ALL 3 algorithms
    set_tf_idf = set(top_tf_idf)
    set_bm25 = set(top_bm25)
    set_custom = set(top_custom)
    
    overlap_all = len(set_tf_idf & set_bm25 & set_custom)
    
    print(f"\n>> Document Overlap in Top {top_k} (All 3 Algorithms): {overlap_all} documents")
    print(f">> Overlap (TF-IDF vs BM25): {len(set_tf_idf & set_bm25)}")
    print(f">> Overlap (TF-IDF vs Custom): {len(set_tf_idf & set_custom)}")
    print(f">> Overlap (BM25 vs Custom): {len(set_bm25 & set_custom)}")
    
    return

def main():
    dataset_file = os.path.join(ROOT_DIR, "data", "fashion_products_dataset.json")
    K = 10 
    #define custom field weights for ranking
    FIELD_WEIGHTS = {
            'title': 4.0,
            'brand': 3.0,
            'sub_category': 2.0,
            'description': 1.0,
            'product_details': 1.0
        }
    print(f"Loading dataset from {dataset_file}...")
    dataset = get_json_data(dataset_file)
        
    print("Building Standard TF-IDF Inverted Index...")
    index, tf, df, idf, document_index = tf_idf.create_index_tfidf(dataset)
    print(f"Standard Index built successfully for {len(document_index)} documents.\n")
    
    
    print("Building Custom Weighted Inverted Index...")
    index_custom, tf_custom, df_custom, idf_custom, document_index_custom = custom.create_index_tfidf_custom(dataset, FIELD_WEIGHTS)
    print(f"Custom Index built successfully for {len(document_index_custom)} documents.\n")

    test_queries = [
            "women tshirt round neck cotton",
            "ecko unl printed sweatshirt",
            "men solid blue track pants",
            "satin necktie set pocket square cufflinks",
            "checkered full sleeve party blazer"
    ]
    
    print("Evaluating queries...")
    for query in test_queries:
        compare_retrival(
            query=query, 
            index=index, 
            idf=idf, 
            tf=tf, 
            document_index=document_index, 
            index_custom=index_custom, 
            idf_custom=idf_custom, 
            tf_custom=tf_custom, 
            document_index_custom=document_index_custom, 
            top_k=K
        )

if __name__ == "__main__":
    main()