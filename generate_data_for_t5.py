import argparse
import csv
import json
import os
import pandas as pd


def create_canard_tsv_from_file_for_t5(input_json_file, output_file, v2=False):
    """
    Create data to from the CANARD dataset to train the T5 model for the query rewriting task
    input_json_file - str, is CANARD's train, dev or test json
    output_file - str, name of the csv output_file
    v2 - bool, if v2 only put [TURN] between turns (query, document) pair else
    always put [TURN] after each query and document
    The generated file is a tsv with 2 columns: first is the context and second is the query rewritten
    """
    with open(input_json_file, "r") as input_file, open(output_file, "w") as out_file:
        input_json = json.load(input_file)
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for query in input_json:
            history = ""
            if v2:  # only put [TURN] between turns (query, document) pair
                for i, turn in enumerate(query["History"]):
                    if i != 0 and i % 2 == 0:
                        history += " [TURN] " + turn
                    else:
                        history += " " + turn + "."
            else:  # always put [TURN] after each query and document
                history = " [TURN] ".join(
                    query["History"])
            question_orig = query["Question"]
            question_rewritten = query["Rewrite"]
            context = question_orig + " [CTX] " + history
            tsv_writer.writerow([context, question_rewritten])


# Example usage using V1 format
# create_canard_training_set_for_t5("./CANARD_Release/train.json", "/t5_training_data/training_t5_canard_data.tsv")
# create_canard_validation_set_for_t5("./CANARD_Release/dev.json","/t5_training_data/validation_t5_canard_data.tsv")
# create_canard_test_set_for_t5("./CANARD_Release/test.json", "/t5_training_data/test_t5_canard_data.tsv")

# Example usage using V2 format
# create_canard_tsv_from_file_for_t5("./CANARD/CANARD_Release/train.json", "/t5_training_data/training_t5_canard_data_v2.tsv", v2=True)
# create_canard_tsv_from_file_for_t5("./CANARD/CANARD_Release/dev.json","/t5_training_data/validation_t5_canard_data_v2.tsv", v2=True)
# create_canard_tsv_from_file_for_t5("./CANARD/CANARD_Release/test.json", "/t5_training_data/test_t5_canard_data_v2.tsv", v2=True)


def create_cast_tsv_from_file_for_t5(input_json_file, resolved_json, output_file):
    """
    Create data to from TREC CAsT's 2019 annotated resolved fiel to train/evaluate the T5 model
    for the query rewriting task
    input_json_file - str, is TREC CAsT's evaluation set
    resolved_json - str, is TREC CAsT's evaluation set with coreferences resolved
    output_file - str, name of the csv output_file
    There is no v2 option because in TREC CAsT 2019 dataset the queries only depend on previous queries
    The generated file is a tsv with 2 columns: first is the context (previous queries)
    and second is the query rewritten
    """
    with open(input_json_file, "r") as input_file, open(resolved_json, "r") as resolved_file, open(output_file,
                                                                                                   "w") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        input_json = json.load(input_file)

        df = pd.read_csv(resolved_file, sep="\t", header=None, index_col=0, names=["turn", "query"])

        for topic in input_json:
            topic_number = topic["number"]
            resolved_utterances_array = []
            for turn in topic["turn"]:
                turn_number = turn["number"]
                topic_turn_id = '%d_%d' % (topic_number, turn_number)

                current_resolved_utterance = df.loc[topic_turn_id]["query"]

                if turn["number"] == 1:
                    tsv_writer.writerow([turn["raw_utterance"], turn["raw_utterance"]])
                else:
                    history = " [TURN] ".join(resolved_utterances_array)
                    question_orig = turn["raw_utterance"]
                    context = question_orig + " [CTX] " + history
                    tsv_writer.writerow([context, current_resolved_utterance])

                resolved_utterances_array.append(current_resolved_utterance)

# Example usage
# create_cast_tsv_from_file_for_t5("./2019_data/evaluation_topics_v1.0.json", "./2019_data/evaluation_topics_annotated_resolved_v1.0.tsv", "/t5_training_data/trec_cast_evaluation.tsv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Generate data for training and evaluation the T5 query rewriting model.''')
    parser.add_argument('--collection', required=True, type=str, help='CANADRD or CAST')
    parser.add_argument('--input_file', required=True, type=str, help='input file json')
    parser.add_argument('--output_file', required=True, type=str, help='output file in tsv format')
    parser.add_argument('--resolved_file', required=False, type=str,
                        default="./2019_data/evaluation_topics_annotated_resolved_v1.0.tsv",
                        help='resolved queries for TREC CAsT in tsv format')
    parser.add_argument('--version', default=2, type=int, required=False, help='1 or 2')

    args = parser.parse_args()

    dir_path = os.path.dirname(args.output_file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if args.collection == "CANARD":
        if args.version not in [1, 2]:
            raise ValueError("The version must be 1 or 2")
        if args.version == 1:
            version_2 = False
        else:
            version_2 = True
        create_canard_tsv_from_file_for_t5(args.input_file, args.output_file, version_2)
    elif args.collection == "CAST":
        create_cast_tsv_from_file_for_t5(args.input_file, args.resolved_file, args.output_file)
    else:
        print("collection not supported try CANARD or CAST")

    print('Done!')


# Example using V1 format
# python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release/train.json> --output_file ./t5_training_data/training_t5_canard_data.tsv --version 1
# python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release/dev.json> --output_file ./t5_training_data/validation_t5_canard_data.tsv  --version 1
# python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release/test.json> --output_file ./t5_training_data/test_t5_canard_data.tsv  --version 1


# Example using V2 format
# python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release/train.json> --output_file ./t5_training_data/training_t5_canard_data_v2.tsv  --version 2
# python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release/dev.json> --output_file ./t5_training_data/validation_t5_canard_data_v2.tsv  --version 2
# python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release/test.json> --output_file ./t5_training_data/test_t5_canard_data_v2.tsv  --version 2


# Example usage using CAsT
# python3 generate_data_for_t5.py --collection CAST --input_file ./2019_data/evaluation_topics_v1.0.json --output_file ./t5_training_data/trec_cast_evaluation.tsv --resolved_file ./2019_data/evaluation_topics_annotated_resolved_v1.0.tsv
