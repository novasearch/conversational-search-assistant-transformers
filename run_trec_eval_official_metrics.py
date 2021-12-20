import argparse
import os
import subprocess
import pandas as pd


# To get the trec_eval script you can follow this link: https://trec.nist.gov/trec_eval/

def run_to_csv_using_trec_eval(run_names, out_file_name,
                               trec_eval_location='./alter_library_code/anserini/eval/trec_eval.9.0.4/trec_eval',
                               path_to_qrels='./2019_data/evaluation_topics_mod.qrel'):
    """
    Writes to tsv file the results obtained for each run using trec_eval script
    run_names - list of strs with path to runs
    out_file_name - str, name of the results file
    trec_eval_location - str, path to the trec eval script
    path_to_qrels - str, path to qrels file
    """
    results = {}
    for run_name in run_names:
        output = subprocess.run([trec_eval_location,
                                 '-m', 'map',
                                 '-m', 'recip_rank',
                                 '-m', 'recall',
                                 '-m', 'all_trec',
                                 '-c', path_to_qrels,
                                 run_name],
                                stdout=subprocess.PIPE).stdout.decode('utf-8')
        # print("output", output)
        current_metrics = {}
        print(f"{os.path.basename(run_name)} Output Metrics:\n {output}")
        print()
        lines = output.split("\n")

        for line in lines[:-1]:  # last is empty line
            metric, _, value = line.split("\t")
            current_metrics[metric.rstrip()] = value  # possible conversion needed to float or int

        # concatenate
        results[os.path.basename(run_name)] = current_metrics

    df = pd.DataFrame.from_dict(results, orient="index")

    # get only some columns
    df1 = df[['recall_1000', 'map', 'recip_rank', 'ndcg_cut_3', 'P_3']]

    # write to file
    df1.to_csv("results/" + out_file_name + ".tsv", sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Get the official metrics using the Trec eval script.''')

    parser.add_argument('--run_name', required=True,
                        type=str, help='Path to file in the trec run format')

    parser.add_argument('--out_file_name', required=True,
                        type=str, help='Path to output file to write the metrics')

    parser.add_argument('--trec_eval_location', required=True,
                        type=str, help='Path to the trec_eval script')

    parser.add_argument('--path_to_qrels', required=True, default='./2019_data/evaluation_topics_mod.qrel',
                        type=str, help='Path to the CAsT qrels file.')

    args = parser.parse_args()

    print("Started running...")
    run_to_csv_using_trec_eval(run_names=[args.run_name], out_file_name=args.out_file_name,
                               trec_eval_location=args.trec_eval_location, path_to_qrels=args.path_to_qrels)

    print('Done!')


# Command examples
# python3 run_trec_eval_official_metrics.py --run_name <path to trec run file> --out_file_name <path to output file> --trec_eval_location ./alter_library_code/anserini/eval/trec_eval.9.0.4/trec_eval --path_to_qrels ./2019_data/evaluation_topics_mod.qrel
