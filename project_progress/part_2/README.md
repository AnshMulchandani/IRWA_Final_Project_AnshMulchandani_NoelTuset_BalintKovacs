# Part 2: Indexing and Evaluation

## Setup

From the repository root:

```
pip install -r requirements.txt

```

Put the dataset at `data/fashion_products_dataset.json` and the labels at `data/validation_labels.csv`.
The NLTK stop word list downloads automatically on first run.

And activate the `irwa.venv`

## Files

| File | What it does |
|---|---|
| `inverted_index.py` | Creates the inverted_index, (`create_index_tfidf`), search for documents that have all the terms in query and returns the documents ranked using TF-IDF.(`search_tf_idf`) the ranking part is done by (`rank_documents`)


| File | What it does |
|---|---|
| `evaluation.py` | Evaluates the search engine's retrieval and ranking performance against ground truth data. Computes depth-specific metrics (`calculate_precision_at_k`, `calculate_recall_at_k`, `calculate_f1_score_at_k`, `calculate_ndcg_at_k`) and overall ranking quality metrics (`calculate_average_precision`, `calculate_reciprocal_rank`). The main function executes this calculation across all queries.


## Run

All commands from `project_progress`:

To compute all the metrics related to ranking and retrival performance:
```python (or python3)
python -m part_2.evaluation
```

## Function Reference (`inverted_index.py`)

- **`create_index_tfidf(data)`**: Builds the inverted index mapping terms to documents and positions, and calculates Term Frequency (TF), Document Frequency (DF), and Inverse Document Frequency (IDF). Returns the inverted index, parallel TF lists, DF, IDF, and the preprocessed document index.
- **`rank_documents(terms, docs, index, idf, tf, document_index)`**: Calculates the cosine similarity (dot product) between the query vector and candidate document vectors using their normalized TF-IDF weights. Returns a sorted list of document IDs from most to least relevant.
- **`search_tf_idf(query, index, idf, tf, document_index)`**: Preprocesses the user's input query, filters for candidate documents that contain all terms in the query (Boolean AND search), and passes them to rank_documents for scoring. Returns the final ranked list.

For `create_index_tfidf`:
1. Preprocess the entire dataset using preprocess_doc to retrieve a clean document_index.
2. Loop through every document in the index and combine text fields (title, description, brand, sub_category, and product_details values) into a single full_text string, then split it into terms.
3. Build a local page index that tracks the term frequencies and the array of positional indices for every word in the current document.
4. Calculate the document's L2 norm (the square root of the sum of squared term frequencies).
5. For each term in the local index, calculate the normalized TF (frequency divided by L2 norm), update the global Document Frequency (DF) counter, and append the document ID and positions to the global inverted index.
6. After processing all documents, compute the IDF for every unique term using the formula log(total_documents / df[term]).

For `rank_documents`:
1. Initialize a query_vector (a list of zeros) and a doc_vectors dictionary (mapping each candidate document ID to a list of zeros). Both vectors are sized to the number of terms in the query.
2. Count the frequency of each term in the query and compute the L2 norm (Euclidean norm) of these frequencies to normalize the query length.
3. Iterate through each term in the query along with its index position. If the term does not exist in the global index, skip it.
4. Calculate the query's TF-IDF score for the current term (Query TF = term count / query L2 norm, multiplied by the term's precomputed IDF) and assign it to the corresponding position in the query_vector.
5. Loop through the inverted index postings for the current term. If the document in the posting is present in the candidate docs list, calculate its TF-IDF score (the document's precomputed TF for that term multiplied by the term's IDF) and update the document's doc_vector at the current term's position.
6. Calculate the final score for each candidate document by computing the dot product (cosine similarity) between its populated doc_vector and the query_vector.
7. Pair each document ID with its computed score and sort the list in descending order so the most relevant documents appear at the top.
8. Extract and return the ordered list of document IDs.

For `search_tf_idf`:
1. Clean the raw query string using preprocess_line. Return an empty list if invalid.
2. Split the cleaned query into individual terms.
3. Iterate through each term and perform a strict conjunctive search (AND) by taking the set intersection (docs.intersection) of all documents containing the current term.
4. Return an empty list if any term is completely missing from the index or if the intersection becomes empty.
5. Pass the resulting candidate documents and the query terms to rank_documents to compute cosine similarity scores and return the ranked list.

## Function Reference (`evaluation.py`)
- **`calculate_precision_at_k(retrieved_docs, relevant_docs, k)`**: Computes the proportion of the top K retrieved documents that are actually relevant. Returns hits / K.
- **`calculate_recall_at_k(retrieved_docs, relevant_docs, k)`**: Computes the proportion of all available relevant documents that were successfully retrieved in the top K. Returns hits / total_relevant_docs.
- **`calculate_f1_score_at_k(p_at_k, r_at_k)`**: Calculates the harmonic mean of Precision@K and Recall@K. Returns 0 if their sum is 0.
- **`calculate_average_precision(retrieved_docs, relevant_docs)`**: Computes Average Precision (AP) for a single query by averaging the precision at every rank where a relevant document is found.
- **`calculate_reciprocal_rank(retrieved_docs, relevant_docs)`**: Computes the Reciprocal Rank (RR) based on the position of the first relevant document retrieved (1 / rank).
- **`calculate_ndcg_at_k(retrieved_docs, relevant_docs, k)`**: Computes the Normalized Discounted Cumulative Gain (NDCG) at K by calculating the discounted gain of the actual ranking and dividing it by the Ideal DCG (IDCG).
- **`main()`**: Orchestrates the evaluation pipeline by loading the dataset, building the index, reading the validation CSV, executing queries, and calculating all cumulative and averaged metrics.

For `main`:
1. Define the absolute paths to the dataset and validation files using os.path, and set the target K value.
2. Load the JSON dataset and call create_index_tfidf to build the TF-IDF inverted index.
3. Load the validation CSV and establish a queries_map dictionary that links the numeric query_ids in the CSV to their corresponding textual search queries.
4. Group the validation dataframe by query_id to evaluate one specific search at a time.
5. For each query group, filter for rows where labels == 1 to extract a list of true relevant document IDs (pid).
6. Execute the search engine (search_tf_idf) using the mapped query text to retrieve a ranked list of candidate documents.
7. Call the individual metric functions (AP, RR, Precision, Recall, F1, NDCG) and add their results to cumulative total variables.
8. After processing all queries, divide the cumulative totals by the number of evaluated queries to print the final averaged metrics (Mean Average Precision, Mean Reciprocal Rank, Mean NDCG@K, etc.).

## AI usage
We used Gemini to review our evaluation and inverted_index code and debug errors in it, to help write some parts of the code like how to do certain computations or certain ranking decisions, and to review our draft explanation of these decisions. We checked the code and the results ourselves and can explain every function.