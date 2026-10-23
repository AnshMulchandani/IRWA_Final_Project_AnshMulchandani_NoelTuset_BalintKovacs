"""
Evidence for the Part 1 field decisions (points 3 and 4). Run from project_progress/part_1:

    python field_analysis.py --data ../../data/fashion_products_dataset.json --labels ../../data/validation_labels.csv

It prints:
  1. how distinctive each candidate field is (point 3)
  2. how many labelled documents each field strategy can reach with a conjunctive (AND)
     query, for the two validation queries (point 3)
  3. what the numeric fields look like once parsed (point 4)
  4. how the spellings of t-shirt are spread (the NORMALISE_TSHIRT rule)
  5. what the numeric fields would look like as text terms (the 'textual form' question)
"""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict

from fields import STRATEGIES, build_document_record, identifier_detail_keys, numeric_terms, parse_numeric_fields, searchable_text
from preprocess import build_terms

def load_products(path):
    """Read the dataset: a list of product dicts."""
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


VALIDATION_QUERIES = {'1': 'women full sleeve sweatshirt cotton', '2': 'men slim jeans blue'}


def field_distinctiveness(data, fields=('category', 'sub_category', 'brand', 'seller')):
    """Distinct values, share of the most common value and its idf, per field."""
    n = len(data)
    print('\n1. Distinctiveness (idf = log10(N / df) of the most common value)')
    for field in fields:
        counts = Counter(product.get(field) or '' for product in data)
        filled = {value: c for value, c in counts.items() if value}
        top, top_count = max(filled.items(), key=lambda item: item[1])
        print(f'  {field:13} distinct {len(filled):4}  most common "{top}" '
              f'{top_count} ({100 * top_count / n:.1f}%)  idf {math.log10(n / top_count):.3f}  empty {counts[""]}')


def conjunctive_matches(index, query):
    """Documents that contain every term of the query (AND semantics, as in Parts 2 and 3)."""
    terms = build_terms(query)
    postings = [index.get(term, set()) for term in terms]
    return set.intersection(*postings) if postings else set()


def compare_strategies(data, labels, skip_keys):
    """For each field strategy: AND matches and labelled documents reached, per validation query."""
    print('\n2. Field strategies on the validation queries (conjunctive match)')
    for name, fields in STRATEGIES.items():
        index = defaultdict(set)
        for product in data:
            for term in set(build_terms(searchable_text(product, skip_keys, fields))):
                index[term].add(product['pid'])
        print(f'  {name:18} vocabulary {len(index)}')
        for qid, query in VALIDATION_QUERIES.items():
            matches = conjunctive_matches(index, query)
            rows = [row for row in labels if row['query_id'] == qid]
            relevant = {row['pid'] for row in rows if row['labels'] == '1'}
            not_relevant = {row['pid'] for row in rows if row['labels'] == '0'}
            print(f'    q{qid} "{query}": {len(matches)} matches, relevant reached '
                  f'{len(relevant & matches)}/{len(relevant)}, non-relevant reached '
                  f'{len(not_relevant & matches)}/{len(not_relevant)}')


def numeric_summary(data):
    """Missing values and ranges of the parsed point 4 fields."""
    print('\n3. Numeric fields once parsed (point 4)')
    parsed = [parse_numeric_fields(product) for product in data]
    for field in ('selling_price', 'actual_price', 'discount', 'average_rating'):
        values = [record[field] for record in parsed if record[field] is not None]
        print(f'  {field:15} missing {len(parsed) - len(values):5}  min {min(values):8.1f}  '
              f'max {max(values):8.1f}  mean {sum(values) / len(values):8.2f}')
    print(f'  out_of_stock    true {sum(record["out_of_stock"] for record in parsed)}')




def textual_form(data):
    """Point 4: the numeric fields as raw text terms versus band terms."""
    print('\n5. Numeric fields as text (point 4)')
    n = len(data)
    for field in ('selling_price', 'actual_price', 'discount', 'average_rating'):
        raw = Counter(product.get(field) or '' for product in data)
        raw.pop('', None)
        print(f'  {field:15} {len(raw):5} distinct raw strings; e.g. "{data[0].get(field)}" '
              f'-> build_terms {build_terms(data[0].get(field) or "")}')
    bands = Counter(term for product in data for term in numeric_terms(build_document_record(product)))
    for term, count in sorted(bands.items()):
        print(f'  band term {term:18} df {count:6}  idf {math.log10(n / count):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='../../data/fashion_products_dataset.json')
    parser.add_argument('--labels', default='../../data/validation_labels.csv')
    args = parser.parse_args()

    data = load_products(args.data)
    with open(args.labels, newline='', encoding='utf-8') as fp:
        labels = list(csv.DictReader(fp))
    skip_keys = identifier_detail_keys(data)
    print(f'{len(data)} products; product_details keys left out of the text: {sorted(skip_keys)}')

    field_distinctiveness(data)
    compare_strategies(data, labels, skip_keys)
    numeric_summary(data)
    textual_form(data)
