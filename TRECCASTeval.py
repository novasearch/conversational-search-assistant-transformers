import json
import numpy as np
import pandas as pd
import rank_metric as metrics


# commands to install to run official trec_cast_eval (in folder where trec_cast_eval is located)
# ./trec_eval -m map -m recip_rank -m recall -m all_trec -c <path_to_qrel_file> <path_to_run_in_trec_format>
# ./trec_eval -m map -m recip_rank -m recall -m all_trec -c 2019_data/evaluation_topics_mod.qrel runs/lmd_T5_1000_2019.run


class ConvSearchEvaluationGeneral:
    """
    Class used to calculate metrics using TREC CAsT dataset
    """

    def __init__(self, topics_json_paths, qrel_file_paths):
        """

        Parameters
        ----------
        topics_json_paths - list of str where each one is a path to the TREC CAsT topics in json format
        qrel_file_paths - list of str where each one is a path to the qrel files for TREC CAsT
        """

        self.topics_json_paths = topics_json_paths
        self.qrel_files_paths = qrel_file_paths

        self.topics = []
        self.relevance_judgments = pd.DataFrame(columns=["topic_turn_id", "dummy", "docid", "rel"])
        for topic in topics_json_paths:
            with open(topic, "rt", encoding="utf-8") as f:
                self.topics += json.load(f)

        for qrel in qrel_file_paths:
            current_qrel = pd.read_csv(qrel, sep=' ', names=["topic_turn_id", "dummy", "docid", "rel"])
            self.relevance_judgments = pd.concat([self.relevance_judgments, current_qrel], ignore_index=True)

        set_of_conversations = set(self.relevance_judgments['topic_turn_id'])
        self.judged_conversations = np.unique([a.split('_', 1)[0] for a in set_of_conversations])

    def eval(self, result, topic_turn_id, only_marco, remove_wapo=True):
        """
        Eval a turn in a given topic
        Parameters
        ----------
        result - pandas dataframe ordered by a some scoring function with column _id,
        topic_turn_id - str, example: topic=31 and turn=3 topic_turn_id must be 31_3
        only_marco - bool, if True only_evaluates using the MS MARCO documents and relevance judgements (keep False)
        remove_wapo - bool, if True removes wapo documents from the dataframe (to use only during testing)
        Returns
        -------
        dict - containing the metrics for that turn
        """

        if remove_wapo:
            result = result.loc[result["_id"].str.contains("WAPO") == False]

        total_retrieved_docs = result.count()[0]

        # get the relevance judgements for the topic_turn_id
        aux = self.relevance_judgments.loc[self.relevance_judgments['topic_turn_id'] == topic_turn_id]

        if np.size(aux) == 0:
            print("No judgements found for topic turn id:", topic_turn_id)

        intersect_retrieved_judged = len(set(result["_id"]) & set(aux["docid"]))
        judged_all = intersect_retrieved_judged / total_retrieved_docs  # how many in all returned were analyzed
        judged_analyzed = intersect_retrieved_judged / len(aux["docid"])  # how many that were analysed found
        judged_top10 = len(set(result["_id"][:10]) & set(aux["docid"])) / 10  # how many in top 10 were analyzed

        if only_marco:  # if we just want to evaluate performance considering only MS MARCO docs
            rel_docs = aux.loc[(aux['rel'] != 0) & (aux['docid'].str.contains("MARCO"))]
        else:
            rel_docs = aux.loc[aux['rel'] != 0]

        query_rel_docs = rel_docs['docid']
        relv_judg_list = rel_docs['rel']
        total_relevant = relv_judg_list.count()

        true_pos = np.intersect1d(result['_id'][:1000], query_rel_docs)  # recall at 1000
        recall = np.size(true_pos) / total_relevant

        true_pos_100 = np.intersect1d(result['_id'][:100], query_rel_docs)  # recall at 100
        recall_100 = np.size(true_pos_100) / total_relevant

        true_pos_10 = np.intersect1d(result['_id'][:10], query_rel_docs)  # recall at 10
        recall_10 = np.size(true_pos_10) / total_relevant

        # Compute vector of results with corresponding relevance level
        relev_judg_results = np.zeros((total_retrieved_docs, 1))
        for index, doc in rel_docs.iterrows():
            relev_judg_results = relev_judg_results + ((result['_id'] == doc.docid) * doc.rel).to_numpy()

        # Normalized Discount Cummulative Gain

        if total_retrieved_docs < 10:  # This was for a very specific case with a query from WAPO
            p10, mrr = 0, 0
        else:
            p10 = metrics.precision_at_k(relev_judg_results[0], 10)
            mrr = metrics.mean_reciprocal_rank(relev_judg_results[0], k=10)

        ndcg5 = metrics.ndcg_at_k(r=relev_judg_results[0], k=5, method=1)
        ap = metrics.average_precision(relev_judg_results[0], total_relevant)

        # NDCG@3, P@1, P@3, MRR e MAP
        p1 = metrics.precision_at_k(relev_judg_results[0], 1)
        p3 = metrics.precision_at_k(relev_judg_results[0], 3)
        ndcg3 = metrics.ndcg_at_k(r=relev_judg_results[0], k=3, method=1)

        return {"recall1000": recall, "recall100": recall_100, "recall10": recall_10, "ap": ap, "mrr": mrr,
                "ndcg5": ndcg5, "ndcg3": ndcg3, "p1": p1, "p3": p3, "p10": p10,
                "judged_all": judged_all, "judged_analyzed": judged_analyzed, "judged_top10": judged_top10}
