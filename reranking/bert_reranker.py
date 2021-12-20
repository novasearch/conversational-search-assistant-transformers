import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class BertReranker:
    """
    Reranker that uses the default bert model with a linear layer on top
    Uses the pooled output of the linear layer given by the model to calculate relevance
    """

    def __init__(self, tokenizer_path="nboost/pt-bert-large-msmarco",
                 model_path="nboost/pt-bert-large-msmarco", use_cuda=False, use_batches=True, batch_size=8):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        self.max_position_embeddings = self.model.config.max_position_embeddings
        self.RERANKER_TYPE = "BERT"
        self.use_cuda = use_cuda
        self.use_batches = use_batches
        self.batch_size = batch_size
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model.cuda()
            else:
                self.device = torch.device("cpu")
                print("No Cuda device found using cpu")
        else:
            self.device = torch.device("cpu")
        self.model.eval()

    def restart_conversation(self):
        """
        Method used to restart some type of memory or state of previous utterances
        """
        pass

    def __input_bert_encoder__(self, row, utterance):
        """
        Outputs BERT embeddings
        Parameters
        ----------
        utterance - str with the query
        row - a row from  a dataframe with columns "_id" (str), "_doc" (str), and "_score" (float)
        Returns
        -------
        tuple - output of BERT model
        """
        inputs = self.tokenizer(utterance, row["_doc"],
                                add_special_tokens=True,
                                max_length=self.max_position_embeddings,
                                truncation="only_second",
                                padding=True, return_tensors="pt")

        if self.device.type == "cuda":
            inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        return output

    def __batch_input_bert_encoder__(self, rows, utterance):
        """
        Get output from bert encoder working with batches (batch size is the len of rows)
        Parameters
        ----------
        utterance - str with the query
        rows - a dataframe with columns "_id" (str), "_doc" (str), and "_score" (float)
        Returns
        -------
        tuple - output of BERT model
        """
        final_outputs = torch.empty(0).to(self.device)
        batch_text_pairs = []
        for _, row in rows.iterrows():  # create the input
            batch_text_pairs.append((utterance, row["_doc"]))

        # iterate through the batches
        for i in range(0, len(batch_text_pairs), self.batch_size):
            inputs = self.tokenizer(batch_text_pairs[i:i + self.batch_size],
                                    add_special_tokens=True,
                                    max_length=self.max_position_embeddings,
                                    truncation="only_second",
                                    padding=True, return_tensors="pt")

            if self.device.type == "cuda":
                inputs = inputs.to(self.device)

            with torch.no_grad():
                output = self.model(**inputs)
                final_outputs = torch.cat([final_outputs, output.logits], dim=0)

        final_outputs = final_outputs.unsqueeze(dim=0)
        return final_outputs

    def rerank_one(self, utterance, result, reranking_limit):
        """
        Re-ranks a row at a time
        Parameters
        ----------
        utterance - str with the query
        result - dataframe with column "_id" (str), "_doc" (str), and "_score" (float)
        reranking_limit - int
        Returns
        -------
        Pandas dataframe with passages ordered by the model and rows "_id" (str), "_doc" (str), and "_score" (float)
        """
        new_rank = []
        max_score_outside_limit = 0
        # since some are not reranked the score may not reflect the rank so it is necessary to do this
        if reranking_limit < len(result):
            max_score_outside_limit = result.iloc[reranking_limit]["_score"]
        for _, row in result[:reranking_limit].iterrows():
            output = self.__input_bert_encoder__(row, utterance)
            s = torch.nn.Softmax(dim=1)
            if self.device == "cuda":
                probs = s(output[0].to("cpu"))
            else:
                probs = s(output[0])
            new_score = probs.tolist()[0][1]
            new_rank.append({"_id": row["_id"], "_score": new_score + max_score_outside_limit,
                             "_doc": row["_doc"]})

        top_n_df = pd.DataFrame(data=new_rank)
        top_n_df = top_n_df.sort_values(by=['_score'], ascending=False)
        new_result = pd.concat([top_n_df, result[reranking_limit:]])

        return new_result

    def rerank_batches(self, utterance, result, reranking_limit):
        """
        Re-ranks using batches

        Parameters
        ----------
        utterance - str with the query
        result - dataframe with column "_id" (str), "_doc" (str), and "_score" (float)
        reranking_limit - int
        Returns
        -------
        Pandas dataframe with passages ordered by the model and rows "_id" (str), "_doc" (str), and "_score" (float)
        """
        max_score_outside_limit = 0
        # since some are not reranked the score may not reflect the rank so it is necessary to do this
        if reranking_limit < len(result):
            max_score_outside_limit = result.iloc[reranking_limit]["_score"]

        output = self.__batch_input_bert_encoder__(result[:reranking_limit], utterance)

        s = torch.nn.Softmax(dim=1)
        if self.device == "cuda":
            probs = s(output[0].to("cpu"))
        else:
            probs = s(output[0])

        new_score = probs[:, 1].tolist()  # get the probability of being relevant (1)

        new_score = [x + max_score_outside_limit for x in new_score]

        result.loc[:reranking_limit - 1, '_score'] = new_score

        new_result = result.sort_values(by=['_score'], ascending=False)

        return new_result

    def rerank(self, utterance, result, reranking_limit=10):
        """
        Rerank result using the model until the reranking limit
        Parameters
        ----------
        utterance - str, query
        result - dataframe with column "_id" (str), "_doc" (str), and "_score" (float)
        turn_number - int
        topic - int
        reranking_limit - int
        Returns
        -------
        Pandas dataframe with passages ordered by the model and rows "_id" (str), "_doc" (str), and "_score" (float)
        """
        if self.use_batches:
            result = self.rerank_batches(utterance, result, reranking_limit=reranking_limit)
        else:
            result = self.rerank_one(utterance, result, reranking_limit=reranking_limit)

        return result
