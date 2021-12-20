import pandas as pd


def reciprocal_rank_fusion(results_lists, k=60):
    """
    Reciprocal rank fusion implementation following 
    "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"

    results_list is a list of dataframes with columns: _id, _score, _doc ordered by score
    k is the hyperparameter
    """

    doc_ranks = {}  # inside there is dic with keys doc (string) and ranks (list of ints)
    for result in results_lists:
        for index, row in result.iterrows():
            # index starts at 0
            if row["_id"] not in doc_ranks:
                doc_ranks[row["_id"]] = {}
                doc_ranks[row["_id"]]["ranks"] = []
            
            doc_ranks[row["_id"]]["doc"] = row["_doc"]
            doc_ranks[row["_id"]]["ranks"].append(index + 1)
    
    rrf_rank = []
    for doc_id, doc_rank in doc_ranks.items():
        rrf_score = 0
        for rank in doc_rank["ranks"]:
            rrf_score += 1 / (k+rank)
        
        rrf_rank.append({"_id": doc_id, "_score": rrf_score, "_doc": doc_rank["doc"]})

    rrf_result = pd.DataFrame(data=rrf_rank)
    rrf_result = rrf_result.sort_values(by=['_score'], ascending=False)

    return rrf_result
