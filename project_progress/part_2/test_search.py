#TEST FILE GENERATED WITH THE HELP OF GEMINI

import json
import os
import sys

#Setup imports
part_2_dir = os.path.dirname(os.path.abspath(__file__))
project_progress_dir = os.path.abspath(os.path.join(part_2_dir, ".."))
root_dir = os.path.abspath(os.path.join(project_progress_dir, ".."))

for path in [project_progress_dir, part_2_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from inverted_index import create_index_tfidf, search_tf_idf

#path to dataset file inside root data/ directory
dataset_path = os.path.join(root_dir, "data", "fashion_products_dataset.json")

print(f"Loading data from: {dataset_path}")

#open and load the dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

print(f"Loaded {len(sample_data)} products successfully!\n")

#build index and run test query
index, tf, df, idf, document_index = create_index_tfidf(sample_data)
print(f"Index built with {len(index)} terms.")

#list of 5 representative queries for project evaluation
test_queries = [
    "women tshirt round neck cotton",
    "ecko unl printed sweatshirt",
    "men solid blue track pants",
    "satin necktie set pocket square cufflinks",
    "checkered full sleeve party blazer"
]

"""test_queries = [
    "women full sleeve sweatshirt cotton", 
    "men slim jeans blue",
]"""


print("\n================ EVALUATING TEST QUERIES ================")
for q in test_queries:
    results = search_tf_idf(q, index, idf, tf, document_index)
    print(f"\nQuery: '{q}' (Total Matches: {len(results)})")
    
    # Display top 20 matching products with titles
    for rank, pid in enumerate(results[:20], start=1):
        title = document_index[pid].get('title', 'No Title')
        print(f"  {rank}. [{pid}] {title}")