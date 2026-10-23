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
| `preprocess.py` | Turns text into terms (`preprocess_line`), preprocesses every product (`preprocess_doc`) including fields and numeric/boolean parsing, and saves the results in a dictionary with key `pid` of each document |

## Run

All commands from `project_progress/part_1`:

Preprocess the whole dataset from Python:

```python
import json
from preprocess import preprocess_doc, get_json_data

data = get_json_data('../../data/fashion_products_dataset.json')
document_index = preprocess_doc(data)
```

## Options & Fields

- **Fields in the searchable text:** `TEXTUAL_FIELDS` includes `title`, `description`, `brand`, and `sub_category`. Product details are also cleaned and integrated.
- **Numeric fields parsing:** Prices (`selling_price`, `actual_price`), `discount`, `average_rating`, and `out_of_stock` are parsed into proper numbers and booleans via specific helper functions (returning `None` if unreadable/missing).

## Preprocessing steps (`preprocess_line` and `preprocess_doc`)

For `preprocess_line`:
1. Check if the input is a valid string; return an empty string if not.
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
3. Call `build_document_record` to extract and clean the 14 output fields, clean product details, and parse numeric/boolean values.
4. Process textual fields and individual product detail values using `preprocess_line`.
5. Map the product's `pid` to this complete record in the `document_index` dictionary and return it.

## AI usage
We used Claude AI and Gemini to review our preprocessing code and debug errors in it, to help write the number parsing code, to compute statistics, and to review our draft explanation of these decisions. We checked the code and the results ourselves and can explain every function.