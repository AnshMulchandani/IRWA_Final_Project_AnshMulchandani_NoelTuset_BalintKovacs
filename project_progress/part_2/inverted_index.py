from part_1.preprocess import preprocess_doc, preprocess_line
from collections import defaultdict
from array import array
import math
import numpy as np

def create_index_tfidf(data):
    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents
    df = defaultdict(int)  # document frequencies of terms in the corpus
    idf = defaultdict(float)
    document_index = {}
    document_index = preprocess_doc(data) #maybe this is done inside a loop

    #should we add the loop of preprocess_doc here?
    #I added the TF-IDF from teory but some modifications are needed like defining terms

    current_page_index = {}

    for position, term in enumerate(terms):  ## terms contains terms in document?
        try:
            # if the term is already in the dict append the position to the corresponding list
            current_page_index[term][1].append(position)
        except:
            # Add the new term as dict key and initialize the array of positions and add the position
            current_page_index[term]=[prod_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] = df[term] + 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

    # Compute IDF following the formula (3) above. HINT: use np.log
    # Note: It is computed later after we know the df.
    num_documents = len(data)
    for term in df:
        idf[term] = np.round(math.log(float(num_documents/df[term])), 4)

    return index, document_index

def search(query, index):
    """
    The output is the list of documents that contain ALL of the query terms.
    So, we will get the list of documents for ALL query term, and take the intersection of them.
    """
    query = preprocess_line(query)
    #in case query is empty
    if not query:
        return []
    #in orther to process the first term of query
    docs = None
    for term in query:
        try:
            # store in term_docs the ids of the docs that contain "term"
            term_docs=[ posting[0] for posting in index[term]]
            # If this is the first term, docs becomes this set of term_docs
            if docs is None:
                docs = term_docs
            else:
                # docs = docs Intersection term_docs
                docs = docs.intersection(term_docs)
            #in case there is no match for the first term of the query we end the loop to optimize the result
            if not docs:
                return []
        except:
            #term is not in index
            pass
    docs = list(docs) if docs else []
    return docs