# Part 1: Text processing and exploratory data analysis

## Setup

From the repository root:

```
pip install -r requirements.txt
```

Put the dataset at `data/fashion_products_dataset.json` and the labels at `data/validation_labels.csv`.
The NLTK stop word list downloads automatically on first run.

## Files

| File | What it does |
|---|---|
| `preprocess.py` | Turns text into terms (`preprocess_line`), cleans details (`clean_product_details`), parses numeric and boolean fields, builds document records (`build_document_record`), and indexes the full dataset (`preprocess_doc`) |

## Run

All commands from `project_progress/part_1`:

Preprocess the whole dataset from Python:

```python (or python3)
import json
from preprocess import preprocess_doc, get_json_data

data = get_json_data('../../data/fashion_products_dataset.json')
document_index = preprocess_doc(data)
```

## Options & Fields

- **Fields in the searchable text:** `TEXTUAL_FIELDS` includes `title`, `description`, `brand`, and `sub_category`. Product details are also cleaned and integrated.
- **Numeric fields parsing:** Prices (`selling_price`, `actual_price`), `discount`, `average_rating`, and `out_of_stock` are parsed into proper numbers and booleans via specific helper functions (returning `None` if unreadable/missing).

## Function Reference (`preprocess.py`)

- **`get_json_data(doc_path)`**: Opens and loads the raw dataset from a JSON file path using UTF-8 encoding and returns the parsed Python object/list.
- **`preprocess_line(line)`**: Performs text normalization on a string input (lowercasing, stripping punctuation/hyphens/apostrophes, splitting, removing NLTK English stop words, applying Porter stemming, and joining back into a space-separated string). Returns an empty structure if input is invalid.
- **`clean_product_details(details)`**: Flattens a list of single-key product detail dictionaries into a single normalized dictionary, preprocessing both keys and values and handling duplicate keys by joining their values.
- **`parse_price(raw)`**: Cleans raw price strings (e.g., removing commas) and converts them into floats. Returns `None` if unreadable or empty.
- **`parse_discount(raw)`**: Extracts numeric percentage values from discount strings using regular expressions, or converts numeric values directly. Returns `None` if unreadable or empty.
- **`parse_rating(raw)`**: Converts raw rating strings into floats, returning `None` if unreadable or empty.
- **`parse_numeric_fields(product)`**: Extracts and parses the non-text fields (`out_of_stock`, `selling_price`, `actual_price`, `discount`, `average_rating`) into correct types (booleans and floats).
- **`build_document_record(product)`**: Constructs the primary 14-field output record dictionary for a single product, applying text preprocessing to textual fields, flattening details, and appending parsed numeric metrics.
- **`preprocess_doc(data)`**: Iterates through the entire dataset, processes each product using `build_document_record`, and maps every product's unique `pid` to its final record in a `document_index` dictionary.

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

## AI usage
We used Claude AI and Gemini to review our preprocessing code and debug errors in it, to help write the number parsing code, to compute statistics, and to review our draft explanation of these decisions. We checked the code and the results ourselves and can explain every function.