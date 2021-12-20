import argparse
import collections
import functools
import operator
import pandas as pd
import os
from pyserini.search import SimpleSearcher
from pyserini.analysis import get_lucene_analyzer
from reranking.bert_reranker import BertReranker
from query_rewriting.query_config import query_configurations_2019
from search.search import union_search, search_anserini
from query_rewriting.query_construction import query_construction
from utils import construct_filename, write_doc_results_to_file, write_trec_results
from fusion.rank_fusion import reciprocal_rank_fusion
from TRECCASTeval import ConvSearchEvaluationGeneral
from tqdm import tqdm


# 2019 tests


# Base retrieval methods and query formats for 2019
def run_topics_general_reranker(trec_cast_eval, searcher, query_config, reranker,
                                reranking_threshold, reranker_query_config=None, use_rrf=False,
                                only_marco=False, remove_wapo=True):
    """
    Calculate the metrics and return the documents
    It also uses a reranker
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    """
    metrics = {}
    all_results = pd.DataFrame()
    _nturns = 0

    for topic in tqdm(trec_cast_eval.topics):
        utterance_vector = []
        conv_id = topic['number']
        if str(conv_id) not in trec_cast_eval.judged_conversations.tolist():
            continue

        title = ""
        if "title" in topic:
            title = topic["title"]

        first_query = ""

        reranker.restart_conversation()

        for turn in topic['turn']:
            turn_id = turn['number']
            if turn_id == 1:
                first_query = turn['raw_utterance']
            utterance = query_construction(raw_utterance=turn['raw_utterance'], conv_id=conv_id, turn_id=turn_id,
                                           query_config=query_config, history=first_query, title=title)
            utterance_vector.append(utterance)
            topic_turn_id = '%d_%d' % (conv_id, turn_id)

            aux = trec_cast_eval.relevance_judgments.loc[
                trec_cast_eval.relevance_judgments['topic_turn_id'] == topic_turn_id]

            if only_marco:
                num_rel = aux.loc[(aux['rel'] != 0) & (aux['docid'].str.contains("MARCO"))]['docid'].count()
            else:
                num_rel = aux.loc[aux['rel'] != 0]['docid'].count()

            if num_rel == 0:
                continue

            print(topic_turn_id)

            if query_config.use_union:
                result = union_search(searcher, query_config.full_union, utterance_vector)
            else:
                result = search_anserini(searcher, utterance)

            # we can use a different query at re-ranking time
            if reranker_query_config:
                reranker_query = query_construction(raw_utterance=turn['raw_utterance'], conv_id=conv_id,
                                                    turn_id=turn_id,
                                                    query_config=reranker_query_config, history=first_query,
                                                    title=title)
                result_reranked = reranker.rerank(reranker_query, result, reranking_threshold)

                if use_rrf:
                    result_reranked = reciprocal_rank_fusion([result, result_reranked])

                result_reranked["query"] = reranker_query
                result_reranked["retrieval_query"] = utterance

            else:
                print("start re-ranking")
                result_reranked = reranker.rerank(utterance, result, reranking_threshold)
                result_reranked["query"] = utterance
                print("end re-ranking")

            tmp_metrics = trec_cast_eval.eval(result_reranked[['_id', '_score']], topic_turn_id, only_marco,
                                              remove_wapo)

            if remove_wapo:  # remove WAPO docs
                result_reranked = result_reranked.loc[result_reranked["_id"].str.contains("WAPO") == False]

            result_reranked["topic_turn_id"] = topic_turn_id

            all_results = all_results.append(result_reranked, ignore_index=True)

            if not metrics:
                metrics = tmp_metrics
            else:
                metrics = dict(functools.reduce(operator.add, map(collections.Counter, [metrics, tmp_metrics])))

            _nturns = _nturns + 1

    return metrics, _nturns, all_results


def run_topics_general_retrieval(trec_cast_eval, searcher, query_config,
                                 only_marco=False, remove_wapo=True):
    """
    Calculate the the metrics and return the documents
    Performs simple retrieval (without any sort of reranking)
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    """
    metrics = {}
    all_results = pd.DataFrame()
    _nturns = 0

    for topic in tqdm(trec_cast_eval.topics):
        utterance_vector = []
        conv_id = topic['number']

        if str(conv_id) not in trec_cast_eval.judged_conversations.tolist():
            continue
        title = ""
        if "title" in topic:
            title = topic["title"]
        first_query = ""

        for turn in topic['turn']:
            turn_id = turn['number']
            if turn_id == 1:
                first_query = turn['raw_utterance']
            utterance = query_construction(raw_utterance=turn['raw_utterance'], conv_id=conv_id, turn_id=turn_id,
                                           query_config=query_config, history=first_query, title=title)
            utterance_vector.append(utterance)
            topic_turn_id = '%d_%d' % (conv_id, turn_id)

            aux = trec_cast_eval.relevance_judgments.loc[
                trec_cast_eval.relevance_judgments['topic_turn_id'] == topic_turn_id]

            if only_marco:
                num_rel = aux.loc[(aux['rel'] != 0) & (aux['docid'].str.contains("MARCO"))]['docid'].count()
            else:
                num_rel = aux.loc[aux['rel'] != 0]['docid'].count()

            if num_rel == 0:
                continue

            if query_config.use_union:
                result = union_search(searcher, query_config.full_union, utterance_vector)
            else:
                result = search_anserini(searcher, utterance)

            tmp_metrics = trec_cast_eval.eval(result[['_id', '_score']], topic_turn_id, only_marco, remove_wapo)

            if remove_wapo:  # remove WAPO docs
                result = result.loc[result["_id"].str.contains("WAPO") == False]
            result["topic_turn_id"] = topic_turn_id
            result["query"] = utterance

            all_results = all_results.append(result, ignore_index=True)

            if not metrics:
                metrics = tmp_metrics
            else:
                metrics = dict(functools.reduce(operator.add, map(collections.Counter, [metrics, tmp_metrics])))

            _nturns = _nturns + 1

    return metrics, _nturns, all_results


def run_topics_general(trec_cast_eval, query_config, searcher, reranker,
                       reranking_threshold=10, reranker_query_config=None, use_rrf=False):
    """
    Run topics in trec_cast_eval
    The function called depends if there is a reranker provided
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    """
    if reranker:
        metrics, _nturns, all_results = \
            run_topics_general_reranker(trec_cast_eval=trec_cast_eval, searcher=searcher,
                                        query_config=query_config, reranker=reranker,
                                        reranking_threshold=reranking_threshold,
                                        reranker_query_config=reranker_query_config, use_rrf=use_rrf)
    else:
        metrics, _nturns, all_results = \
            run_topics_general_retrieval(trec_cast_eval=trec_cast_eval, searcher=searcher,
                                         query_config=query_config)

    for k, value in metrics.items():
        metrics[k] = value / _nturns

    print(metrics)

    return metrics, _nturns, all_results


def run_test_general_base_retrieval_methods(query_dic, query_types, trec_cast_eval, similarity, string_params,
                                            searcher: SimpleSearcher, reranker,
                                            write_to_trec_eval, write_results_to_file, reranker_query_config,
                                            reranking_threshold, use_rrf):
    """
    Run topics in trec_cast_eval
    query_dic is a dict with string keys and a QueryConfig object
    query_types is a list of strings that denote keys we want to use that are in query_dic
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    If write_to_trec_eval writes the results in trec eval format
    If write_results_to_file writes the results in tsv format (including the
    query and document's content for later use)
    """
    metric_results = {}
    doc_results = {}

    for query_type in query_types:
        print(similarity + " " + query_type + " " + string_params)
        current_key = similarity + "_" + query_type + "_" + string_params

        metric_results[current_key], _, doc_results[current_key] = \
            run_topics_general(trec_cast_eval=trec_cast_eval,
                               query_config=query_dic[query_type], searcher=searcher,
                               reranker=reranker,
                               reranker_query_config=reranker_query_config,
                               reranking_threshold=reranking_threshold,
                               use_rrf=use_rrf)

        index_name = os.path.basename(os.path.normpath(searcher.index_dir))
        run_file_name = index_name + "_" + current_key
        run_name = query_type
        if searcher.is_using_rm3():
            run_file_name += "_rm3"
        if reranker:
            run_file_name += "_" + reranker.RERANKER_TYPE + "_" + str(reranking_threshold)
            run_name += "_" + reranker.RERANKER_TYPE + "_" + str(reranking_threshold)

        if write_to_trec_eval:
            write_trec_results(file_name=run_file_name, result=doc_results[current_key],
                               run_name=run_name)
        if write_results_to_file:
            write_doc_results_to_file(file_name=run_file_name + ".tsv", result=doc_results[current_key])

    return metric_results, doc_results


# pre-built method to use
def run_general_example(trec_cast_eval=ConvSearchEvaluationGeneral(["./2019_data/evaluation_topics_v1.0.json"],
                                                                   ["./2019_data/evaluation_topics_mod.qrel"]),
                        similarity="lmd",
                        index="./car_marco_wapo",
                        custom_name="", rm3=None, reranker=None,
                        write_to_trec_eval=False, write_results_to_file=False, query_dict=query_configurations_2019,
                        reranker_query_config=None,
                        reranking_threshold=None, use_rrf=False):
    """
    Pre-built example to run query variants in TREC CAsT data
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    RM3 is a dict containing keys fd_docs, fd_terms, orig_query_weight
    """

    searcher = SimpleSearcher(index)
    searcher.set_analyzer(get_lucene_analyzer(stemmer="krovetz"))
    type_queries = list(query_dict.keys())

    if rm3:
        searcher.set_rm3(fb_docs=rm3["fd_docs"], fb_terms=rm3["fd_terms"],
                         original_query_weight=rm3["orig_query_weight"])
    if similarity == "lmd":
        i = 1000
        similarity_params = [i]
        searcher.set_qld(mu=1000)
        metric_results, doc_results = \
            run_test_general_base_retrieval_methods(query_dict, type_queries, trec_cast_eval=trec_cast_eval,
                                                    searcher=searcher, similarity="lmd",
                                                    string_params=str(i), reranker=reranker,
                                                    write_to_trec_eval=write_to_trec_eval,
                                                    write_results_to_file=write_results_to_file,
                                                    reranker_query_config=reranker_query_config, use_rrf=use_rrf,
                                                    reranking_threshold=reranking_threshold)
    elif similarity == "bm25":
        i = 1.1  # my fine tunning 1.1 clark 1.2
        j = 0.3  # my fine tunning 0.3 clark 0.75
        similarity_params = [i, j]
        searcher.set_bm25(k1=i, b=j)
        metric_results, doc_results = \
            run_test_general_base_retrieval_methods(query_dict, type_queries, trec_cast_eval=trec_cast_eval,
                                                    searcher=searcher, similarity="bm25",
                                                    string_params=str(i) + "_" + str(j), reranker=reranker,
                                                    write_to_trec_eval=write_to_trec_eval,
                                                    write_results_to_file=write_results_to_file,
                                                    reranker_query_config=reranker_query_config, use_rrf=use_rrf,
                                                    reranking_threshold=reranking_threshold)
    else:
        raise ValueError("similarity must be one of these: lmd, bm25")

    df = pd.DataFrame.from_dict(metric_results, orient='index')
    type_reranker = None
    if reranker:
        type_reranker = reranker.RERANKER_TYPE
    if rm3:
        rm3_params = "fd_{}_ft_{}_qw_{}".format(rm3["fd_docs"], rm3["fd_terms"], rm3["orig_query_weight"])

        file_name = construct_filename(False, False, similarity, similarity_params, index, custom_name, rm3_params,
                                       type_reranker, reranking_threshold)
    else:
        file_name = construct_filename(False, False, similarity, similarity_params, index, custom_name, "",
                                       type_reranker, reranking_threshold)
    df.to_csv("results/" + file_name + ".csv")
    return df, metric_results, doc_results


def run_test_general_from_file(trec_cast_eval, reranker, filename,
                               ranking_docs_file_name=None, utterances=None, reranking_threshold=10,
                               write_to_trec_eval=False, write_results_to_file=False):
    """
    Run reranking method on files
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    ranking_docs_file_name is a tsv files with columns
    0=index, 1=_id, 2=_score, 3=_doc, 4=_topic_turn_id, 5=query
    utterance is a dictionary of utterances to use in reranking, if none is specified the queries in the file are used
    """

    test_ranking_docs = pd.read_csv(ranking_docs_file_name, sep="\t", index_col=0)
    metrics, n_turns, all_results = run_topics_general_reranker_from_file(
        trec_cast_eval=trec_cast_eval,
        ranking_docs=test_ranking_docs,
        reranker=reranker,
        utterances=utterances,
        reranking_threshold=reranking_threshold,
        use_rrf=False)

    if write_to_trec_eval:
        write_trec_results(file_name=filename, result=all_results, run_name=filename)
    if write_results_to_file:
        write_doc_results_to_file(file_name=filename + ".tsv", result=all_results)

    df = pd.DataFrame.from_dict(metrics, orient='index')
    df.to_csv("results/" + filename + ".csv")

    return metrics, all_results


def run_topics_general_reranker_from_file(trec_cast_eval, ranking_docs, reranker,
                                          utterances, reranking_threshold, use_rrf=False, remove_wapo=True):
    """
    Run reranking method on files
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    ranking_docs - pandas dataframe with columns _id, _score, _doc, topic_turn_id and query
    utterance - dictionary of utterances to use in reranking, if none is specified is used the queries in the file
    """
    metrics = {}
    all_results = pd.DataFrame()
    _nturns = 0
    for topic in trec_cast_eval.topics:
        conv_id = topic['number']
        if str(conv_id) not in trec_cast_eval.judged_conversations.tolist():
            continue
        reranker.restart_conversation()
        for turn in topic['turn']:
            turn_id = turn['number']
            topic_turn_id = '%d_%d' % (conv_id, turn_id)
            # print(topic_turn_id)

            aux = trec_cast_eval.relevance_judgments.loc[
                trec_cast_eval.relevance_judgments['topic_turn_id'] == topic_turn_id]
            num_rel = aux.loc[aux['rel'] != 0]['docid'].count()

            if num_rel == 0:
                continue

            result = ranking_docs.loc[(ranking_docs['topic_turn_id'] == topic_turn_id)]
            if remove_wapo:  # remove WAPO docs
                result = result.loc[result["_id"].str.contains("WAPO") == False]

            # when using batches it is necessary to reset_index
            result.reset_index(drop=True, inplace=True)

            if not result.empty:
                if utterances:
                    reranker_query = utterances[str(conv_id)][str(turn_id)]  # use a new query for reranking
                else:
                    reranker_query = result['query'].iloc(0)[0]  # get the query used to generate first ranking

                result_reranked = reranker.rerank(reranker_query, result, reranking_threshold)
                result_reranked["query"] = reranker_query
                result_reranked["topic_turn_id"] = topic_turn_id

                if use_rrf:
                    result_reranked = reciprocal_rank_fusion([result, result_reranked])

                tmp_metrics = trec_cast_eval.eval(result_reranked[['_id', '_score']], topic_turn_id, False)

            else:
                print("topic_turn_id not found in file")
                continue

            all_results = all_results.append(result_reranked, ignore_index=True)

            if not metrics:
                metrics = tmp_metrics
            else:
                metrics = dict(functools.reduce(operator.add, map(collections.Counter, [metrics, tmp_metrics])))

            _nturns = _nturns + 1

    for k, value in metrics.items():
        metrics[k] = value / _nturns

    return metrics, _nturns, all_results


def run_file_cast_general_to_metrics_csv(trec_cast_eval, run_name, out_filename=None, remove_wapo=True):
    """
    Calculates the evaluation metrics on a run from CAsT file using our code
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    out_filename - string, name of file to store the csv file with the metrics results if not
    specified the file gets the same name as run_name
    run_name - string, run file in TREC format
    remove_wapo - bool, remove wapo files present in run
    """
    data = pd.read_csv(run_name, sep=" ", header=None)
    data.columns = ["topic_turn_id", "dummy", "_id", "rank", "_score", "run_name"]
    list_topics = data["topic_turn_id"].unique()

    metrics = {}
    _nturns = 0

    for topic_turn_id in list_topics:
        aux = trec_cast_eval.relevance_judgments.loc[
            trec_cast_eval.relevance_judgments['topic_turn_id'] == topic_turn_id]
        num_rel = aux.loc[aux['rel'] != 0]['docid'].count()
        topic_turn_id_data = data.loc[data["topic_turn_id"] == topic_turn_id]

        if num_rel == 0:
            continue

        tmp_metrics = trec_cast_eval.eval(topic_turn_id_data[['_id', '_score']], topic_turn_id, False, remove_wapo)

        if not metrics:
            metrics = tmp_metrics
        else:
            metrics = dict(functools.reduce(operator.add, map(collections.Counter, [metrics, tmp_metrics])))

        _nturns = _nturns + 1

    if out_filename is None:
        out_filename = os.path.basename(os.path.normpath(run_name))

    final_metrics = {}
    for k, value in metrics.items():
        final_metrics[k] = value / _nturns

    df = pd.DataFrame.from_dict({out_filename: final_metrics}, orient='index')
    df.to_csv("results/" + out_filename + ".csv")

    print(final_metrics)
    print()

    return df, final_metrics


def run_file_cast_general_to_metrics_by_turn_depth(trec_cast_eval, run_name, out_filename=None, remove_wapo=True):
    """
    Calculates the evaluation metrics on a run from CAsT file by turn depth
    trec_cast_eval - object of type ConvSearchEvaluationGeneral
    out_filename - string, name of file to store the csv file with the metrics results if not
    specified the file gets the same name as run_name
    run_name - string, run file in TREC format
    remove_wapo - bool, remove wapo files present in run
    """
    data = pd.read_csv(run_name, sep=" ", header=None)
    data.columns = ["topic_turn_id", "dummy", "_id", "rank", "_score", "run_name"]
    list_topics = data["topic_turn_id"].unique()
    turn_depth_metrics = {}
    metrics = {}

    for topic_turn_id in list_topics:
        topic, turn = topic_turn_id.split("_")
        aux = trec_cast_eval.relevance_judgments.loc[
            trec_cast_eval.relevance_judgments['topic_turn_id'] == topic_turn_id]
        num_rel = aux.loc[aux['rel'] != 0]['docid'].count()
        topic_turn_id_data = data.loc[data["topic_turn_id"] == topic_turn_id]

        if num_rel == 0:
            continue

        tmp_metrics = trec_cast_eval.eval(topic_turn_id_data[['_id', '_score']], topic_turn_id, False, remove_wapo)

        if not metrics:
            metrics = tmp_metrics
        else:
            metrics = dict(functools.reduce(operator.add, map(collections.Counter, [metrics, tmp_metrics])))

        if turn not in turn_depth_metrics:
            turn_depth_metrics[turn] = ({}, 0)  # tuple with metrics, number turns

        turn_depth_metrics[turn] = (dict(functools.reduce(operator.add,
                                                          map(collections.Counter,
                                                              [turn_depth_metrics[turn][0], tmp_metrics]))),
                                    turn_depth_metrics[turn][1] + 1)

    if out_filename is None:
        out_filename = os.path.basename(os.path.normpath(run_name))

    final_metrics_by_turn = __reduce_metrics_by(turn_depth_metrics)
    df = pd.DataFrame.from_dict(final_metrics_by_turn, orient='index')

    df.to_csv("results/" + out_filename + ".csv")

    print("final_metrics by turn", final_metrics_by_turn)
    print()

    return df, final_metrics_by_turn


def __reduce_metrics_by(metrics):
    """
    Auxiliary function used to reduce the metrics by the keys in a dictionary
    Example reduce by turn or by query_type
    metrics - dict with keys as the reduce type and each value is a tuple (dict_metrics, number_turns),
    dic_metrics is a dict with the name of the metric as key and a value
    """
    final_metrics = {}
    for turn, value in metrics.items():
        if turn not in final_metrics:
            final_metrics[turn] = {}
        for k, v in value[0].items():  # value[0] is the metrics
            final_metrics[turn][k] = v / value[1]  # value[1] is the number turns
        final_metrics[turn]["number_tuns"] = value[1]
    return final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Run the retrieval and reranking models over TREC CAsT and generate runs and metrics.''')

    parser.add_argument('--topics_json_path', required=False, default="./2019_data/evaluation_topics_v1.0.json",
                        type=str, help='TREC CAsT topics in json format')
    parser.add_argument('--qrel_file_path', required=False, default="./2019_data/evaluation_topics_mod.qrel",
                        type=str, help='TREC CAsT qrels file')

    parser.add_argument('--similarity', required=False, default="lmd",
                        type=str, help='Similarity to use while searching. Allowed BM25 and LMD.')

    parser.add_argument('--index', required=True, default="./car_marco_wapo",
                        type=str, help='Path to the index folder creator using Anserini.')

    parser.add_argument('--custom_name', required=False, default="", type=str, help='Custom name for the file.')

    parser.add_argument('--rm3', required=False, action='store_true', help='Use RM3 before reranking.')

    parser.add_argument('--reranker', required=False, action='store_true', help='Use BERT reranker model.')

    parser.add_argument('--reranking_threshold', required=False, type=int, default=1000,
                        help='Reranking threshold for the reranker model.')

    parser.add_argument('--reranker_batch_size', required=False, type=int,
                        default=8, help='Batch size of the BERT reranker model.')

    parser.add_argument('--use_rrf', required=False, action='store_true', help='Use Reciprocal Rank Fusion.')

    args = parser.parse_args()

    reranker_model = None
    if args.reranker:
        print("using BERT Reranker")
        reranker_model = BertReranker(use_cuda=True, use_batches=True, batch_size=args.reranker_batch_size)

    print("Started running...")
    run_general_example(trec_cast_eval=ConvSearchEvaluationGeneral([args.topics_json_path], [args.qrel_file_path]),
                        similarity=args.similarity,
                        index=args.index,
                        custom_name=args.custom_name, rm3=args.rm3, reranker=reranker_model,
                        write_to_trec_eval=True, write_results_to_file=False,
                        query_dict=query_configurations_2019,
                        reranker_query_config=None,
                        reranking_threshold=args.reranking_threshold, use_rrf=args.use_rrf)

    print('Done!')


# Command examples

# using retrieval only
# python3 run_test_generalizable.py --topics_json_path ./2019_data/evaluation_topics_v1.0.json --qrel_file_path ./2019_data/evaluation_topics_mod.qrel --similarity lmd --index ./car_marco_wapo

# using reranker
# python3 run_test_generalizable.py --topics_json_path ./2019_data/evaluation_topics_v1.0.json --qrel_file_path ./2019_data/evaluation_topics_mod.qrel --similarity lmd --index ./car_marco_wapo --reranker
