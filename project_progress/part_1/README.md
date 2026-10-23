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
| `preprocess.py` | Turns text into terms (`build_terms`) and preprocesses every product (`preprocess_doc`) |
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

Every product keeps its original fields (title and description stay readable for the results) and gets:

- `processed_title`, `processed_description`: the terms of the title and of the description
- `processed_terms`: the terms of every searchable field (title, description, brand, sub_category, product_details values), the input for the Part 2 index
- `product_details` as one dictionary (`{'Fabric': 'Cotton Blend', ...}`) instead of a list
- `selling_price`, `actual_price`, `discount`, `average_rating` as numbers, `None` when missing; `out_of_stock` as a boolean

## Options

- **Fields in the searchable text:** `SEARCHABLE_FIELDS` in `fields.py` (title, description, brand, sub_category, product_details values). The strategies compared in the report are in `STRATEGIES`.
- **T-shirt spellings:** `NORMALISE_TSHIRT` in `preprocess.py` (default `True`) maps t-shirt, t shirt and tshirt to one term.
- **Codes left out of the text:** `identifier_detail_keys(data, max_unique_share=0.5)` in `fields.py`.
- **Numeric fields as band terms:** `preprocess_doc(data, numeric_as_terms=True)` adds terms like `price_500_999` and `in_stock` (`numeric_terms` in `fields.py`). Off by default; the report explains why.

## Preprocessing steps (`build_terms`)

1. Lowercase.
2. Map the t-shirt spellings to `tshirt` (if `NORMALISE_TSHIRT`).
3. Delete hyphens and apostrophes, so `t-shirt` gives `tshirt` and `men's` gives `mens`.
4. Replace every other punctuation mark with a space, so `look.Designed` gives two words.
5. Split on whitespace.
6. Remove NLTK English stop words.
7. Porter stemming.

The query must go through the same `build_terms`.
