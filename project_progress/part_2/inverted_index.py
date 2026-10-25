from part_1.preprocess import preprocess_doc, preprocess_line
from collections import defaultdict
from array import array

def create_index(data):
    index = defaultdict(list)
    document_index = {}
    document_index = preprocess_doc(data)

    current_page_index = {}
    for position, term in enumerate(terms): #term is title + description
        try:
            # if the term is already in the index for the current page (current_page_index)
            # append the position to the corresponding list
             current_page_index[term][1].append(position)
        except:
            # Add the new term as dict key and initialize the array of positions and add the position
            current_page_index[term]=[prod_pid, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(current_page_index[term_page])

    return index, document_index