# Open-Domain Conversational Search Assistant with Transformers

This is the original repository for the paper Open-Domain Conversational Search Assistant with Transformers 
available [here](https://arxiv.org/pdf/2101.08197.pdf).

## Getting Started
Install/Clone Anserini (Java) following this [link](https://github.com/castorini/anserini).
  * You will also need Java 11

Install the rest of the necessary dependencies with the package manager of your choice.
If you use conda you can create the env using the **search_assistant_env.yml** file
(depending on the hardware some versions might need to be different):

`conda env create -f search_assistant_env.yml`

Or you can manually install the necessary packages:
* pandas - `pip install pandas` or `conda install pandas` 
* numpy - `pip install numpy` or `conda install numpy`
* cbor - `pip install cbor`
* torch - `pip install torch torchvision torchaudio` or `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
* transformers - `pip install transformers` or `conda install -c huggingface transformers`
* faiss-cpu - `pip install faiss-cpu` or `conda install faiss-cpu -c pytorch`
* pyserini - `pip install pyserini` or [conda version](https://github.com/castorini/pyserini/blob/master/docs/installation.md)


## Data

### Trec CAsT data
The Topics and Resolved Topic Annotations from TREC CAsT 2019 are already provided in the folder 2019_data.

Please follow the original location of the data [here](https://www.treccast.ai/) in the **Year 1 (TREC 2019)** part to get 
the documents to index.
The download links for the raw data are also provided below but follow the original [link](https://www.treccast.ai/) 
for a more comprehensive explanation.
* Link to [MS MARCO](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz) and [MS MARCO duplicate files](http://boston.lti.cs.cmu.edu/Services/treccast19/duplicate_list_v1.0.txt)
* Link to [TREC CAR](http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz)
* Link to [WAPO](https://ir.nist.gov/wapo/WashingtonPost.v2.tar.gz) and [WAPO duplicate files](http://boston.lti.cs.cmu.edu/Services/treccast19/wapo_duplicate_list_v1.0.txt)

## Processing and Indexing the Data
Instead of preprocessing and indexing the data yourself, you can also download our 
anserini/pyserini index from [here](https://drive.google.com/file/d/1QT2yNoLune-1x_QViIEM9wnjOvrGqWcB/view?usp=sharing) (~30 GB).

### Preprocessing
After downloading the data we parse all sources to the same jsonl representation 
(it also removes the duplicates):

`python3 convert_collections_to_jsonl.py --collection "MARCO" --collection_path <.../collection.tsv> --output_folder <path_to_output_folder> --duplicates_file <path to duplicates file>/duplicate_list_v1.0_MARCO.txt` 

`python3 convert_collections_to_jsonl.py --collection "CAR" --collection_path <.../paragraphCorpus/dedup.articles-paragraphs.cbor> --output_folder <path_to_output_folder>`

`python3 convert_collections_to_jsonl.py --collection "WAPO" --collection_path <.../wapo/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl> --output_folder <path_to_output_folder> --duplicates_file <path to duplicates file>/wapo_duplicate_list_v1.0.txt`

This process can take some time. 
If you are using the default parameters you should have a folder with **51 json files** in the end.

### Indexing
Then we use [Anserini](https://github.com/castorini/anserini) to index and search the data.
To index the data created in the previous step run **only one** of the commands below:

Index using Anserini:

`sh ./anserini/target/appassembler/bin/IndexCollection -collection JsonCollection  -generator LuceneDocumentGenerator -threads 2 -input <path to jsonl files folder>  -index <.../index_output_location>/car_marco_wapo -storePositions -storeDocvectors -storeRaw -stemmer krovetz`


Index using pyserini:

`python -m pyserini.index --input <path to jsonl files folder> --collection JsonCollection --generator LuceneDocumentGenerator --index <.../index_output_location>/car_marco_wapo --threads 2 --storePositions --storeDocvectors --storeRaw --stemmer krovetz`



## Query Rewriting Model
The query rewriting model in the paper is based on a T5 model trained on the 
[CANARD](https://sites.google.com/view/qanta/projects/canard) dataset.

The model trained on the CANARD is available to download [here](https://drive.google.com/file/d/1TBWNWHSxFYzDIZ8wVbFXKMQRWSrAfUq0/view?usp=sharing). 
Instructions on how to load and use the model are provided in the
python notebook available at: **colab_notebooks/t5_query_rewriter.ipynb**.

### Processed Queries
If you are only interested in the queries already processed by the query rewriting model
you can skip the training of the model and use the queries available at:
* coreferenced_resolved_files/trec_cast_complete_t5_real_time_v1.json 
  
* coreferenced_resolved_files/trec_cast_complete_t5_real_time_v2.json 


### Training the Model

#### Creating the training and evaluation data
The preprocessed data to train the T5 model is already available in the folder T5_training_data, but if
you want to create the data start by downloading the [CANARD](https://sites.google.com/view/qanta/projects/canard) dataset and unzip it to a folder.

After this run the following commands to obtain the data to train the model 
(the V1 and V2 formats are explained in the paper):
* Using V1 format:

`python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release>/train.json --output_file ./t5_training_data/training_t5_canard_data.tsv --version 1`

`python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release>/dev.json --output_file ./t5_training_data/validation_t5_canard_data.tsv  --version 1`

`python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release>/test.json --output_file ./t5_training_data/test_t5_canard_data.tsv  --version 1`
 

* Using V2 format:

`python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release>/train.json --output_file ./t5_training_data/training_t5_canard_data_v2.tsv  --version 2`

`python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release>/dev.json --output_file ./t5_training_data/validation_t5_canard_data_v2.tsv  --version 2`

`python3 generate_data_for_t5.py --collection CANARD --input_file <.../CANARD_Release>/test.json --output_file ./t5_training_data/test_t5_canard_data_v2.tsv  --version 2`


To generate the input for the T5 model using the TREC CAsT dataset (only for evaluation purposes):

`python3 generate_data_for_t5.py --collection CAST --input_file ./2019_data/evaluation_topics_v1.0.json --output_file ./t5_training_data/trec_cast_evaluation.tsv --resolved_file ./2019_data/evaluation_topics_annotated_resolved_v1.0.tsv`


#### Training the model
To train the model load the python notebook available at: **colab_notebooks/t5_query_rewriter.ipynb**
into Google Colab and follow the steps detailed there. 

You will need a Google cloud storage account, and you can use the free credits provided by Google to train the model on TPUs.

Create a bucket on your account, 
load the training data to that bucket and put the bucket name in the notebook 
in the places where it is needed.

The colab notebook is a modified version of the original notebook provided by the T5 creators.
Follow the instructions on the [original repository](https://github.com/google-research/text-to-text-transfer-transformer) if in need of any additional information.

At the end of the notebook execution, you should have a trained T5 model for query rewriting.

After this just feed the model the queries from TREC CAsT and gather the outputs 
to use in the retrieval and reranking steps.


## Create the Runs
After having the queries generated by the T5 model, we can now evaluate the proposed architecture 
composed of the transformer query rewriter, the retrieval model, and the transformer reranker on the TREC CAsT 2019 dataset.

As explained in the paper the query rewriter is a T5 model, the retrieval model is LMD, 
and the transformer reranker is a BERT Model trained on MS MARCO from [here](https://huggingface.co/nboost/pt-bert-base-uncased-msmarco). 

The TREC CAsT 2019 runs for the evaluation set are already provided in the **runs** folder, 
however, you can also create them by following the rest of this section.

### Retrieval Only
To generate the retrieval only runs for all query types run:

`python3 run_test_generalizable.py --topics_json_path ./2019_data/evaluation_topics_v1.0.json --qrel_file_path ./2019_data/evaluation_topics_mod.qrel --similarity lmd --index <.../index_output_location>/car_marco_wapo`

### Reranking
To generate the retrieval and reranking runs for all query types run 
(it is highly recommended to use a GPU device to create this run):

`python3 run_test_generalizable.py --topics_json_path ./2019_data/evaluation_topics_v1.0.json --qrel_file_path ./2019_data/evaluation_topics_mod.qrel --similarity lmd --index <.../index_output_location>/car_marco_wapo --reranker --reranker_batch_size 8`

### Results
The run_test_generalizable.py script will generate various **.run** files available at the **runs** folder, 
and various **.csv** files in the **results** folder with the **unnoficial** metrics.

## Evaluate the Runs
The metrics outputted from the run_test_generalizable.py script are not the official metrics, so we now run the
official [trec_eval](https://trec.nist.gov/trec_eval/) script over the generated runs.
We provide a version of trec_eval but you can also download a newer version of 
trec_eval from [here](https://trec.nist.gov/trec_eval/) if needed.

If you download a newer version of trec_eval you may need to change some files in the official script to include the metrics at rank 3.
* trec_eval.9.0.4/m_map_cut.c
* trec_eval.9.0.4/m_P.c
* trec_eval.9.0.4/m_rel_P.c
* trec_eval.9.0.4/m_ndcg_cut.c

Just add the desired ranks to the **long_cutoff_array variable**, e.g.:

`static long long_cutoff_array[] = {1, 3, 5, 10, 15, 20, 30, 100, 200, 500, 1000}`

Run the command to compile again inside the trec_eval directory:

`cd trec_eval.9.0.4 && make`

Finally, run this command with every **.run** file generated by run_test_generalizable.py 
to get the official results which will be written to the **results folder**: 

`python3 run_trec_eval_official_metrics.py --run_name <path to trec run file> --out_file_name <name of output file> --trec_eval_location ./trec_eval.9.0.4/trec_eval --path_to_qrels ./2019_data/evaluation_topics_mod.qrel`


## Citation
If you find anything useful please cite our work using:
```
@inproceedings{DBLP:conf/ecir/FerreiraLSM21,
  author    = {Rafael Ferreira and
               Mariana Leite and
               David Semedo and
               Jo{\~{a}}o Magalh{\~{a}}es},
  editor    = {Djoerd Hiemstra and
               Marie{-}Francine Moens and
               Josiane Mothe and
               Raffaele Perego and
               Martin Potthast and
               Fabrizio Sebastiani},
  title     = {Open-Domain Conversational Search Assistant with Transformers},
  booktitle = {Advances in Information Retrieval - 43rd European Conference on {IR}
               Research, {ECIR} 2021, Virtual Event, March 28 - April 1, 2021, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12656},
  pages     = {130--145},
  publisher = {Springer},
  year      = {2021},
  url       = {https://doi.org/10.1007/978-3-030-72113-8\_9},
  doi       = {10.1007/978-3-030-72113-8\_9},
  timestamp = {Wed, 07 Apr 2021 16:01:38 +0200},
  biburl    = {https://dblp.org/rec/conf/ecir/FerreiraLSM21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
