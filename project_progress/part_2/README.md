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
| `inverted_index.py` | Creates the inverted_index, 
(`create_index_tfidf`), search for documents that have all the terms in query and returns the documents ranked using TF-IDF.
(`search_tf_idf`) the ranking part is done by
(`rank_documents`)
|

| File | What it does |
|---|---|
| `evaluation.py` | Evaluates the search engine's retrieval and ranking performance against ground truth data. Computes depth-specific metrics 
(`calculate_precision_at_k`, `calculate_recall_at_k`, `calculate_f1_score_at_k`, `calculate_ndcg_at_k`) and overall ranking quality metrics (`calculate_average_precision`, `calculate_reciprocal_rank`). The main function executes this calculation across all queries.
|

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

For `search_tf_idf`:
1. Clean the raw query string using preprocess_line. Return an empty list if invalid.
2. Split the cleaned query into individual terms.
3. Iterate through each term and perform a strict conjunctive search (AND) by taking the set intersection (docs.intersection) of all documents containing the current term.
4. Return an empty list if any term is completely missing from the index or if the intersection becomes empty.
5. Pass the resulting candidate documents and the query terms to rank_documents to compute cosine similarity scores and return the ranked list.

## Preprocessing steps (`preprocess_line` and `preprocess_doc`)

For `preprocess_line`:
1. Check if the input is a valid string; return an empty structure if not.
2. Lowercase the string.
3. Delete apostrophes and hyphens (e.g., `t-shirt` becomes `tshirt`, `men's` becomes `mens`).
4. Replace every other punctuation mark with a space.
5. Split on whitespace into tokens.
6. Remove NLTK English stop words.
7. Apply Porter stemming to each token.
8. Join the processed tokens back into a single space-separated string.

For `preprocess_doc`:
1. Loop through every product in the dataset.
2. Extract the product `pid`.
3. Call `build_document_record` to extract and clean the output fields, clean product details, and parse numeric/boolean values.
4. Map the product's `pid` to this complete record in the `document_index` dictionary and return it.

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