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
| `preprocess.py` | Turns text into terms (`build_terms`), preprocesses every product (`preprocess_doc`) and saves the results in a dictionary with key `pid` of each document|
| `fields.py` | Points 2, 3 and 4: the document record, which fields go into the searchable text, parsing of prices, discount, rating and stock |
| `field_analysis.py` | Prints the evidence behind the field decisions: field distinctiveness, field strategies on the two validation queries, numeric field summary, t-shirt spellings |

## Run

All commands from `project_progress/part_1`:

```
python field_analysis.py --data ../../data/fashion_products_dataset.json --labels ../../data/validation_labels.csv
```

Preprocess the whole dataset from Python:

```python
import json
from preprocess import preprocess_doc
with open('../../data/fashion_products_dataset.json', encoding='utf-8') as fp:
    data = json.load(fp)
data = preprocess_doc(data)
```

## Options

- **Fields in the searchable text:** `SEARCHABLE_FIELDS` in `fields.py` (title, description, brand, sub_category, product_details values). The strategies compared in the report are in `STRATEGIES`.
- **Codes left out of the text:** `identifier_detail_keys(data, max_unique_share=0.5)` in `fields.py`.
- **Numeric fields as band terms:** `preprocess_doc(data, numeric_as_terms=True)` adds terms like `price_500_999` and `in_stock` (`numeric_terms` in `fields.py`). Off by default; the report explains why.

## Preprocessing steps (`build_terms` and `preprocess_doc`)

For `build_terms`:
0. Check if string is not null
1. Lowercase.
2. Map the t-shirt spellings to `tshirt` (if `NORMALISE_TSHIRT`).
3. Delete hyphens and apostrophes, so `t-shirt` gives `tshirt` and `men's` gives `mens`.
4. Replace every other punctuation mark with a space, so `look.Designed` gives two words.
5. Split on whitespace.
6. Remove NLTK English stop words.
7. Porter stemming.

The query must go through the same `build_terms`.

For `preprocess_doc`:
1. Check the dataset using identifier_detail_keys to identify product_details keys that are mostly unique.
2. Loop through every product in dataset.
3. Call `build_document_record` to extract and clean the 14 output fields into a base record.
4. Execute `build_terms` on the product's title and save the tokens to `processed_title`.
5. Execute `build_terms` on the product's description and save the tokens to `processed_description`.
6. Stores the preprocessed searchable text (`title`, `description`, `brand`, `sub_category`, and `product_details`) using `searchable_text`, run `build_terms` on it, and save to    `searchable_tokens`.
8. (Optional) Run `numeric_terms` (`selling_price`, `actual_price`, `discount`, `average_rating`)to append numeric terms to `searchable_tokens`.
9. Map the product's pid to this complete record in the `document_index` dictionary and return it.

## AI usage
We used Claude AI and Gemini to review our preprocessing code and debug errors in it, to help write the field handling and number parsing code, to compute the dataset statistics behind points 3 and 4, and to review our draft explanation of these decisions. We checked the code and the results ourselves and can explain every function.