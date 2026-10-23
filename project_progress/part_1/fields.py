"""
Part 1, points 2, 3 and 4: how each field of a product enters the search engine.

Point 2: every document keeps the 14 output fields in a dictionary (the record).
Point 3: category, sub_category, brand, product_details and seller.
    We use a hybrid. The words that tell products apart (brand, sub_category and
    the product_details values) are added to the searchable text, so a
    conjunctive (AND) query can match them. Every field is also kept as a
    structured value in the record, so later parts can filter and weight by field.
    category is not added to the text: 96.6% of products share one value, so it
    carries almost no information for ranking. seller is kept for filtering only.
Point 4: out_of_stock, selling_price, discount, actual_price and average_rating.
    They are parsed into numbers and booleans and never indexed as text terms.
    A missing value stays None (not 0), so it is never mistaken for a real value.
"""
import re
from collections import Counter, defaultdict

#the fields every search result must return (point 2)
OUTPUT_FIELDS = ['pid', 'title', 'description', 'brand', 'category', 'sub_category',
                 'product_details', 'seller', 'out_of_stock', 'selling_price',
                 'discount', 'actual_price', 'average_rating', 'url']

#fields whose words go into the searchable text (point 3)
SEARCHABLE_FIELDS = ['title', 'description', 'brand', 'sub_category', 'product_details']

#fields kept as exact values for filtering (point 3)
FILTER_FIELDS = ['category', 'sub_category', 'brand', 'seller']

#field strategies compared in field_strategy_check.py (point 3)
STRATEGIES = {
    'title_description': ['title', 'description'],
    'all_merged': ['title', 'description', 'brand', 'category', 'sub_category',
                   'product_details', 'seller'],
    'hybrid': SEARCHABLE_FIELDS,
}


def flatten_product_details(details):
    """
    Turn product_details, a list of one-key dicts, into one dict.

    Example: [{'Fabric': 'Cotton'}, {'Pattern': 'Solid'}] -> {'Fabric': 'Cotton', 'Pattern': 'Solid'}
    27 products repeat a key; their values are joined with ', ' so nothing is lost.
    """
    if isinstance(details, dict):
        return dict(details)  #already flattened
    flat = {}
    for entry in details or []:
        for key, value in entry.items():
            flat[key] = value if key not in flat else flat[key] + ', ' + value
    return flat


def identifier_detail_keys(data, max_unique_share=0.5):
    """
    Find the product_details keys whose values are mostly unique, like 'Style Code'.

    A key qualifies when its number of distinct values is more than max_unique_share
    of the times it appears. Such values are codes, not words a user would type,
    so they are left out of the searchable text.
    """
    count, values = Counter(), defaultdict(set)
    for product in data:
        for key, value in flatten_product_details(product.get('product_details')).items():
            count[key] += 1
            values[key].add(value)
    return {key for key in count if len(values[key]) / count[key] > max_unique_share}


def details_text(details, skip_keys):
    """Join the product_details values (not the keys) into one string, skipping identifier keys."""
    return ' '.join(value for key, value in flatten_product_details(details).items()
                    if key not in skip_keys)


def searchable_text(product, skip_keys, fields=SEARCHABLE_FIELDS):
    """Build the text that gets tokenised and indexed for one product (point 3)."""
    parts = []
    for field in fields:
        if field == 'product_details':
            parts.append(details_text(product.get('product_details'), skip_keys))
        else:
            parts.append(product.get(field) or '')
    return ' '.join(part for part in parts if part)


def parse_price(raw):
    """'2,999' -> 2999.0. Empty or unreadable -> None."""
    try:
        return float(str(raw).replace(',', '').strip())
    except ValueError:
        return None


def parse_discount(raw):
    """'69% off' -> 69.0. Empty or unreadable -> None."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)  #already parsed
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', str(raw or ''))
    return float(match.group(1)) if match else None


def parse_rating(raw):
    """'3.9' -> 3.9. Empty or unreadable -> None."""
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def parse_numeric_fields(product):
    """Parse the five non-text fields of point 4 into numbers and a boolean."""
    return {
        'out_of_stock': bool(product.get('out_of_stock', False)),
        'selling_price': parse_price(product.get('selling_price')),
        'actual_price': parse_price(product.get('actual_price')),
        'discount': parse_discount(product.get('discount')),
        'average_rating': parse_rating(product.get('average_rating')),
    }


def numeric_terms(record):
    """
    The 'textual form' alternative for point 4, kept as an option and off by default.

    Each numeric field becomes one coarse band term, e.g. price_500_999, rating_4_plus,
    discount_50_plus, in_stock. Bands keep a little order that raw strings lose
    ('921' and '920' would be unrelated terms, and the punctuation rule would even
    split '2,999' into '2' and '999'). A missing value gives no term.
    """
    terms = ['out_of_stock' if record['out_of_stock'] else 'in_stock']
    price = record['selling_price']
    if price is not None:
        terms.append('price_under_500' if price < 500 else 'price_500_999' if price < 1000
                     else 'price_1000_1999' if price < 2000 else 'price_2000_plus')
    rating = record['average_rating']
    if rating is not None:
        terms.append('rating_4_plus' if rating >= 4 else 'rating_3_4' if rating >= 3 else 'rating_under_3')
    discount = record['discount']
    if discount is not None:
        terms.append('discount_50_plus' if discount >= 50 else 'discount_under_50')
    return terms


def build_document_record(product):
    """
    The record stored for one product (point 2): the 14 output fields, with
    product_details as a dict and the point 4 fields parsed into numbers.
    """
    record = {field: product.get(field) or '' for field in OUTPUT_FIELDS}
    record['product_details'] = flatten_product_details(product.get('product_details'))
    record.update(parse_numeric_fields(product))
    return record


def build_filter_index(document_index, fields=FILTER_FIELDS):
    """
    Exact-value lookup for filtering: {field: {value: set of pids}}.

    Example: filter_index['brand']['York'] -> every pid sold under the brand York.
    """
    filter_index = {field: defaultdict(set) for field in fields}
    for pid, record in document_index.items():
        for field in fields:
            if record[field]:
                filter_index[field][record[field]].add(pid)
    return filter_index
