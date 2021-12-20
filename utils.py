import os
import json
import pandas as pd
import csv
import numpy as np
import torch
import random
from pyserini.search import SimpleSearcher


# General functions  to be used by other modules


def create_prety_json(path_file, out_file):
    """
    Indent json file to make it more readable
    """
    with open(path_file, "r") as json_file, open(out_file, "w") as out:
        data = json.load(json_file)
        json.dump(data, out, indent=4)


def write_trec_results(file_name, result, run_name):
    """
    Write results of the search in a trec run format in the folder runs
    for later evaluation with trec eval
    First column is topic number
    Second column is the query number within that topic.  This is currently unused and should always be Q0
    Third column is the document number of the retrieved document
    Fourth column is the rank the document is retrieve
    Fifth column shows the score (integer or floating point) that generated the ranking - MUST be in descending order
    Sixth column is the run name and it must be 12 or fewer letters and numbers, and no punctuation
    """
    if len(run_name) > 12:
        print("Run name is to big for TREC CAsT, it must be 12 characters or fewer with no punctuation")
    with open("runs/" + file_name + ".run", "w") as res_file:
        for index, row in result.iterrows():
            res_file.write(row["topic_turn_id"] + " Q0 " + row["_id"] + " " + str(index + 1) + " " + str(row["_score"])
                           + " " + str(run_name) + "\n")


def ranking_docs_to_run(ranking_docs_name, run_file_name, run_name):
    """
    Write results of ranking_docs to trec run format
    """
    results = pd.read_csv("ranking_docs/" + ranking_docs_name, sep="\t", header=0)
    # results = pd.read_csv(ranking_docs_name, sep="\t", header=0)
    write_trec_results(run_file_name, results, run_name)


def change_run_name(run, out_run, new_name):
    """
    Change the run name on a file in the trec eval format
    Creates a new file with out_name with the same values as in run but with the last column
    run name as the new name
    """
    with open(run, "r") as r:
        with open(out_run, "w") as w:
            for line in r:
                trec_line = line.split()
                trec_line[-1] = new_name
                w.write(" ".join(trec_line) + "\n")


def write_doc_results_to_file(file_name, result):
    """
    Write results of the search in a tsv format in the folder ranking_docs
    for later use in reranking tasks
    """
    result.to_csv("ranking_docs/" + str(file_name), index=True, sep="\t")


def construct_filename(train, test, similarity, similarity_params, index, custom_filename="", rm3_params="",
                       reranker="", reranking_threshold=""):
    """
    Generate a filename based on the parameters provided for a consistent naming convention
    """
    filename_params = []
    if train:
        filename_params.append("train")
    if test:
        filename_params.append("test")
    filename_params.append(similarity)
    for param in similarity_params:
        filename_params.append(str(param))
    filename_params.append(os.path.basename(os.path.normpath(index)))
    if rm3_params:
        filename_params.append(rm3_params)
    if reranker:
        filename_params.append(reranker)
    if reranking_threshold:
        filename_params.append(str(reranking_threshold))
    if custom_filename:
        filename_params.append(custom_filename)
    return "anserini_" + "_".join(filename_params)


def trec_cast_2019_queries_to_my_format(input_json_file, output_file):
    """
    Generates files in my format from TREC CAsT 2019 format
    Parameters
    ----------
    input_json_file - name of the file with the queries in TREC CAsT's 2019 format
    output_file - name of the file to store the queries in their new format
    """
    with open(input_json_file, "r") as input_file, open(output_file, "w") as out_file:
        output_dic = {}
        input_json = json.load(input_file)
        for topic in input_json:
            topic_number = topic["number"]
            for turn in topic["turn"]:
                turn_number = turn["number"]
                if turn_number == 1:
                    output_dic[str(topic_number)] = {}
                output_dic[str(topic_number)][str(turn_number)] = turn["raw_utterance"]
        json.dump(output_dic, out_file, indent=2)


# Example usage
# create_json_queries_in_my_format("./2019_data/train_topics_v1.0.json", "./2019_data/train_topics_my_format.json")
# create_json_queries_in_my_format("./2019_data/evaluation_topics_v1.0.json", "./2019_data/evaluation_topics_my_format.json")


def evaluated_resolved_2019_to_json():
    """
    Writes 2019 trec cast evaluated resolved data (tsv) into the format used by the program
    """
    dic = {}
    with open('./2019_data/evaluation_topics_annotated_resolved_v1.0.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        for row in reader:
            topic_turn_id = row[0].split("_")
            topic = topic_turn_id[0]
            turn = topic_turn_id[1]
            query = row[1]
            if topic not in dic:
                dic[topic] = {}
            dic[topic][turn] = query

    with open('coreference_resolved_files/original_resolved.json', 'w') as outfile:
        json.dump(dic, outfile, indent=2)


def set_seeds(seed_val=42):
    """
    Sets the seeds of random, numpy and torch to allow for reproducible results
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)


def run_file_to_ranking_docs(run, index,
                             out_filename=None, queries_json=None):
    """
    Create rankink docs tsv file from run file for later use in reranking tasks
    out_filename - string, name of file to store the csv file with the metrics results if not
    specified the file gets the same name as run_name
    run - string, run file in TREC format
    queries_json - string, path to json file using the dic format
    index - string, location of index to search for the documents
    """
    searcher = SimpleSearcher(index)
    _queries = None
    if queries_json is not None:
        with open(queries_json) as queries_file:
            _queries = json.load(queries_file)

    if out_filename is None:
        out_filename = os.path.basename(os.path.normpath(run)) + ".tsv"

    with open(run, "r") as run_file, open("ranking_docs/" + out_filename, "w") as out:
        tsv_writer = csv.writer(out, delimiter='\t')
        tsv_writer.writerow(["", "_id", "_score", "_doc", "topic_turn_id", "query"])  # header
        for i, line in enumerate(run_file):
            query = ""
            topic_turn_id, _, doc_id, rank, score, run_name = line.split()
            document_content = searcher.doc(docid=doc_id)
            document_content = document_content.raw()  # get the raw representation
            if _queries is not None:
                topic, turn = topic_turn_id.split("_")
                query = _queries[topic][turn]
            if document_content:
                document_content = document_content.replace('\n', " ")
                tsv_writer.writerow([i, doc_id, score, document_content, topic_turn_id, query])  # -1 because we
            else:
                print("document with id " + doc_id + " not found")
