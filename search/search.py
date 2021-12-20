import numpy as np
import pandas as pd
from pyserini.search import SimpleSearcher


def search_anserini(searcher: SimpleSearcher, utterance, n_docs=1000):
    """
    Search collection using searcher and utterance provided
    searcher - Anserini searcher
    utterance - str, query
    return result - pandas dataframe with columns _id (str) _score (float) and _doc (str)
    """
    hits = searcher.search(utterance, k=n_docs)
    scores, doc_ids, raw_docs = [], [], []
    for _, hit in enumerate(hits):
        scores.append(hit.score)
        doc_ids.append(hit.docid)
        raw_docs.append(hit.raw)
            
    result = pd.DataFrame({"_id": doc_ids, "_score": scores, "_doc": raw_docs})
    return result


def generate_queries_union(uterance_vector):
    """
    Performs a union of the current query with each of the previous turns separately
    uterance_vector - list, queries in the order they were made
    returns - list of queries in union
    """
    if len(uterance_vector) > 1:
        generate_queries = [utterance + " " + uterance_vector[-1] for utterance in uterance_vector[:-1]]
        return generate_queries
    else:
        return uterance_vector


def union_search(searcher, full_union, utterance_vector, n_docs=1000):
    """
    Using the utterance vector performs union search (multiple queries) if full_union is False
    and performs full-union queries if full_union is true (only one but possibly large query)
    searcher - Anserini searcher
    full_union - boolean, indicates if uses full_union or union
    uterance_vector - list, queries in the order they were made
    return result - pandas dataframe with columns _id (str) _score (float) and _doc (str)
    """
    if full_union:
        generated_queries = [" ".join(utterance_vector)]
    else:
        generated_queries = generate_queries_union(utterance_vector)

    result_dic = {}
    qids = list(map(str, np.arange(len(generated_queries))))

    # perform the queries in parallel to improve efficiency
    hits = searcher.batch_search(queries=generated_queries, qids=qids, k=n_docs, threads=4)
    for hit in hits.values():
        for _, h in enumerate(hit):
            if h.docid not in result_dic:
                result_dic[h.docid] = [h.score, h.raw]
            else:
                # take the max score of each combination of queries
                result_dic[h.docid][0] = max(h.score, result_dic[h.docid][0])

    result_list = []
    for k, v in result_dic.items():
        # k is doc id, v[0] is score, v[1] is doc
        result_list.append((k, v[0], v[1]))
    result = pd.DataFrame(result_list, columns=["_id", "_score", "_doc"])
    result = result.sort_values(by=['_score'], ascending=False)

    # if not full union the amount of documents retrieved is more than n_docs so get only first n_docs
    return result[:n_docs] 
