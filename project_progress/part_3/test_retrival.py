from part_1.preprocess import get_json_data, get_csv_data
import part_2.inverted_index as tf_idf
import part_2.evaluation as eval
import part_3.ranking_bm25 as bm25
import part_3.custom_ranking as custom
import os
import math
import pandas as pd

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

def evaluate_algorithm(algo_name, val_file_path, queries_map, search_function, search_args, k=10):
    """
    Reads the specific validation file, runs the provided search function, 
    and computes cumulative metrics for the algorithm.
    """
    print(f"\n{'='*50}\nEvaluating: {algo_name}\n{'='*50}")
    
    
    val_df = get_csv_data(val_file_path)

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_f1 = 0.0
    cumulative_ndcg = 0.0
    cumulative_ap = 0.0
    cumulative_rr = 0.0
    num_queries = 0

    for query_id, group in val_df.groupby('query_id'):
        if query_id not in queries_map:
            print(f"Warning: No text provided in queries_map for query_id {query_id}. Skipping.")
            continue
            
        query_text = queries_map[query_id]
        relevant_docs = group[group['labels'] == 1]['pid'].tolist()
        
        if not relevant_docs:
            continue
            
        # Unpack the specific arguments needed for this search function
        retrieved_docs = search_function(query_text, *search_args)
        
        # Calculate metrics
        p_at_k = eval.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
        r_at_k = eval.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
        f1_at_k = eval.calculate_f1_score_at_k(p_at_k, r_at_k)
        ap_at_k = eval.calculate_average_precision_at_k(retrieved_docs, relevant_docs, k)
        rr = eval.calculate_reciprocal_rank(retrieved_docs, relevant_docs)
        ndcg_at_k = eval.calculate_ndcg_at_k(retrieved_docs, relevant_docs, k)
        
        cumulative_precision += p_at_k
        cumulative_recall += r_at_k
        cumulative_f1 += f1_at_k
        cumulative_ap += ap_at_k
        cumulative_rr += rr
        cumulative_ndcg += ndcg_at_k
        num_queries += 1

    if num_queries == 0:
        print("No valid queries found for evaluation.")
        return None

    # Calculate and store means
    results = {
        'Algorithm': algo_name,
        f'P@{k}': cumulative_precision / num_queries,
        f'R@{k}': cumulative_recall / num_queries,
        f'F1@{k}': cumulative_f1 / num_queries,
        'MAP': cumulative_ap / num_queries,
        'MRR': cumulative_rr / num_queries,
        f'NDCG@{k}': cumulative_ndcg / num_queries
    }
    
    # Print individual algorithm summary
    for metric, value in results.items():
        if metric != 'Algorithm':
            print(f"{metric+':':<10} {value:.4f}")
            
    return results

def run_full_comparison(K, queries_map, dataset_file, val_file_tfidf, val_file_bm25, val_file_custom):

    print(f"Loading dataset from {dataset_file}...")
    dataset = get_json_data(dataset_file)
    
    # 2. Build Standard Index (used by TF-IDF and BM25)
    index, tf, df, idf, document_index = tf_idf.create_index_tfidf(dataset)
    
    # 3. Build Custom Index (used by Custom Ranker)
    FIELD_WEIGHTS = {'title': 4.0, 'brand': 3.0, 'sub_category': 2.0, 'description': 1.0, 'product_details': 1.0}
    idx_cust, tf_cust, df_cust, idf_cust, doc_idx_cust = custom.create_index_tfidf_custom(dataset, FIELD_WEIGHTS)
    
    # 4. Evaluate TF-IDF
    tfidf_results = evaluate_algorithm(
        algo_name="TF-IDF with Cosine Similarity",
        val_file_path=val_file_tfidf,
        queries_map=queries_map,
        search_function=tf_idf.search_tf_idf,
        search_args=(index, idf, tf, document_index),
        k=K
    )
    
    # 5. Evaluate BM25
    bm25_results = evaluate_algorithm(
        algo_name="BM25",
        val_file_path=val_file_bm25,
        queries_map=queries_map,
        search_function=bm25.search_bm25,
        search_args=(index, document_index), # BM25 doesn't need tf/idf dicts
        k=K
    )
    
    # 6. Evaluate Custom Ranker
    custom_results = evaluate_algorithm(
        algo_name="Custom Search",
        val_file_path=val_file_custom,
        queries_map=queries_map,
        search_function=custom.search_tf_idf_custom,
        search_args=(idx_cust, idf_cust, tf_cust, doc_idx_cust),
        k=K
    )
    
    # 7. Print Final Comparison Table
    print(f"\n{'='*75}\nFINAL ALGORITHM COMPARISON (Top {K})\n{'='*75}")
    all_results = [r for r in [tfidf_results, bm25_results, custom_results] if r is not None]
    
    if all_results:
        df_comparison = pd.DataFrame(all_results).set_index('Algorithm')
        # Multiply by 100 for cleaner percentage-like formatting, or leave as decimals
        print(df_comparison.round(4))

def main():
    dataset_file = os.path.join(ROOT_DIR, "data", "fashion_products_dataset.json")
    validation_file_tf_idf = os.path.join(ROOT_DIR, "project_progress", "part_2", "validation_labels_tf_idf.csv")
    validation_file_bm25 = os.path.join(ROOT_DIR, "project_progress", "part_3", "validation_labels_bm25.csv")
    validation_file_custom = os.path.join(ROOT_DIR, "project_progress", "part_2", "validation_labels_custom.csv")

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
    
    print("Evaluating overlap...")
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
    print("Evaluating performance...")

    queries_map = {
        1: "women tshirt round neck cotton",
        2: "ecko unl printed sweatshirt",
        3: "men solid blue track pants",
        4: "satin necktie set pocket square cufflinks",
        5: "checkered full sleeve party blazer"
    }
    run_full_comparison(
        K=K, 
        queries_map=queries_map, 
        dataset_file=dataset_file, 
        val_file_tfidf=validation_file_tf_idf, 
        val_file_bm25=validation_file_bm25, 
        val_file_custom=validation_file_custom
    )
    

if __name__ == "__main__":
    main()