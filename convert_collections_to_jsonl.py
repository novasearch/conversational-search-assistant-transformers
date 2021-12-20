import json
import os
import argparse
from trec_car.read_data import iter_paragraphs
from urllib import parse
import re

# Functions to convert MARCO, TREC CAR and WAPO to jsonl to later indexing through anserini
# Example indexing commands (in the Anserini Folder):
# sh ./target/appassembler/bin/IndexCollection -collection JsonCollection  -generator LuceneDocumentGenerator -threads 1 -input <path_folder_with_jsonl_files>  -index <path_to_folder_to_save_index> -storePositions -storeDocvectors -storeRaw -stemmer krovetz

# change content of folders (jsonl files) before indexing
# sh ./target/appassembler/bin/IndexCollection -collection JsonCollection  -generator LuceneDocumentGenerator -threads 2 -input ./collection_jsonl  -index ./Data/trec_cast_collection/marco -storePositions -storeDocvectors -storeRaw -stemmer krovetz
# sh ./target/appassembler/bin/IndexCollection -collection JsonCollection  -generator LuceneDocumentGenerator -threads 2 -input ./collection_jsonl  -index ./Data/trec_cast_collection/car_marco -storePositions -storeDocvectors -storeRaw -stemmer krovetz
# sh ./target/appassembler/bin/IndexCollection -collection JsonCollection  -generator LuceneDocumentGenerator -threads 2 -input ./collection_jsonl  -index ./Data/trec_cast_collection/car_marco_wapo -storePositions -storeDocvectors -storeRaw -stemmer krovetz


def parse_sim_file(filename):
    """
    Reads the deduplicated documents file and stores the
    duplicate passage ids into a dictionary
    Portion of code from https://github.com/grill-lab/trec-cast-tools
    """
    sim_dict = {}
    counter = 0
    lines = open(filename).readlines()
    for line in lines:
        data = line.strip().split(':')
        if len(data[1]) > 0:
            sim_docs = data[-1].split(',')
            for docs in sim_docs:
                sim_dict[docs] = 1
        counter += 1
        if counter % 1000000 == 0:
            print(counter)

    return sim_dict


def parse_sim_file_wapo(filename):
    """
    Reads the deduplicated documents file and stores the
    duplicate passage ids into a dictionary
    Portion of code from https://github.com/grill-lab/trec-cast-tools
    """
    # Creates a dict for duplicates for easy access
    dup_dict = {}
    print("dic start")
    data_dups = open(filename).readlines()
    for each in data_dups:
        idxs = each.strip().split(':')
        if len(idxs[-1]) > 0:
            all_idxs = idxs[-1].split(',')
            for every in all_idxs:
                dup_dict[every] = 1
    print("dic end")

    return dup_dict


def convert_collection_marco(f_args):
    """
    Convert MS MARCO collection to many jsonl files
    """
    print('Converting collection...')
    sim_dic = parse_sim_file(f_args.duplicates_file)
    file_index = 0
    with open(f_args.collection_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            doc_id, doc_text = line.rstrip().split('\t')

            if i % f_args.max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = os.path.join(f_args.output_folder, 'docs_marco{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                file_index += 1
            if "MARCO_" + doc_id not in sim_dic:
                output_dict = {'id': 'MARCO_' + doc_id, 'contents': doc_text}
                output_jsonl_file.write(json.dumps(output_dict) + '\n')

            if i % 100000 == 0:
                print('Converted {} docs in {} files'.format(i, file_index))

    output_jsonl_file.close()


def convert_collection_trec_car(f_args):
    """
    Convert TREC CAR collection to many jsonl files
    """
    print('Converting collection...')
    output_jsonl_file = None
    file_index = 0
    i = 0
    for para in iter_paragraphs(open(f_args.collection_path, 'rb')):
        doc_id = "CAR_" + para.para_id
        doc_text = para.get_text()

        if i % f_args.max_docs_per_file == 0:
            if i > 0:
                output_jsonl_file.close()
            output_path = os.path.join(f_args.output_folder, 'docs_car{:02d}.json'.format(file_index))
            output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
            file_index += 1

        output_dict = {'id': doc_id, 'contents': doc_text}
        output_jsonl_file.write(json.dumps(output_dict) + '\n')

        if i % 100000 == 0:
            print('Converted {} docs in {} files'.format(i, file_index))
        i += 1

    output_jsonl_file.close()


def convert_collection_wapo(f_args):
    """
    Convert WAPO collection to many jsonl files
    """
    print('Converting collection...')
    sim_dic = parse_sim_file_wapo(f_args.duplicates_file)
    file_index = 0
    i = 0
    with open(f_args.collection_path, encoding="utf-8") as jl_file:
        for line in jl_file:
            content = json.loads(line.strip())
            main_id = "WAPO_" + str(content["id"])
            paras = content['contents']
            counter = 0
            for para in paras:
                if para is None:
                    continue
                elif "subtype" in para:
                    if para["subtype"] == "paragraph":
                        doc_content = para["content"]
                        counter += 1
                        para_id = main_id + "-" + str(counter)

                        if i % f_args.max_docs_per_file == 0:
                            if i > 0:
                                output_jsonl_file.close()
                            output_path = os.path.join(f_args.output_folder, 'docs_wapo{:02d}.json'.format(file_index))
                            output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                            file_index += 1

                        if para_id not in sim_dic:
                            output_dict = {'id': para_id, 'contents': doc_content}
                            output_jsonl_file.write(json.dumps(output_dict) + '\n')

                        if i % 100000 == 0:
                            print('Converted {} docs in {} files'.format(i, file_index))
                        i += 1

    output_jsonl_file.close()


def sanitize_string(s):
    """
    Clean the string for the metadata dict
    Portion of code from https://github.com/grill-lab/trec-cast-tools
    """
    s = parse.unquote(s)
    s = re.sub(r'\W+', ' ', s)
    s = s.replace("enwiki", "")
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Converts MSMARCO, CAR or WAPO collection to Anserini jsonl files.''')
    parser.add_argument('--collection', required=True, help='MARCO, CAR or WAPO')
    parser.add_argument('--collection_path', required=True, help='collection file')
    parser.add_argument('--output_folder', required=True, help='output file')
    parser.add_argument('--duplicates_file', required=False, help='output file', default="")
    parser.add_argument('--max_docs_per_file', default=1000000, type=int,
                        help='maximum number of documents in each jsonl file.')

    args = parser.parse_args()
    print("Arguments:", args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.collection == "MARCO":
        convert_collection_marco(args)
    elif args.collection == "CAR":
        convert_collection_trec_car(args)

    elif args.collection == "WAPO":
        convert_collection_wapo(args)
    else:
        print("collection not supported try MARCO, CAR or WAPO")

    print('Done!')

# Command examples
# python3 convert_collections_to_jsonl.py --collection "MARCO" --collection_path ./Data/collection.tsv --output_folder ./Data/collection_jsonl --duplicates_file ./Data/duplicate_list_v1.0_MARCO.txt
# python3 convert_collections_to_jsonl.py --collection "CAR" --collection_path ./Data/paragraphCorpus/dedup.articles-paragraphs.cbor --output_folder ./Data/collection_jsonl
# python3 convert_collections_to_jsonl.py --collection "WAPO" --collection_path ./Data/wapo/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl --output_folder ./Data/collection_jsonl --duplicates_file ./Data/wapo/wapo_duplicate_list_v1.0.txt
