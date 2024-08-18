import nltk
import pandas as pd
import numpy as np
# nltk.download('wordnet')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score, f1_score
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.logger import logger 
from typing import Dict, Tuple

class PerformanceMetric: 
    def __init__(self):
        super(PerformanceMetric, self).__init__()


    def _wup_measure(self, a,b,similarity_threshold=0.925):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        def _get_semantic_field(a):
            weight = 1.0
            semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
            return (semantic_field,weight)


        def _get_stem_word(a):
            """
            Sometimes answer has form word\d+:wordid.
            If so we return word and downweight
            """
            weight = 1.0
            return (a,weight)


        global_weight=1.0

        (a,global_weight_a)=_get_stem_word(a)
        (b,global_weight_b)=_get_stem_word(b)
        global_weight = min(global_weight_a,global_weight_b)

        if a==b:
            # they are the same
            return 1.0*global_weight

        if a==[] or b==[]:
            return 0


        interp_a,weight_a = _get_semantic_field(a)
        interp_b,weight_b = _get_semantic_field(b)

        if interp_a == [] or interp_b == []:
            return 0

        # we take the most optimistic interpretation
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if local_score > global_max:
                    global_max=local_score

        # we need to use the semantic fields and therefore we downweight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score=global_max*weight_a*weight_b*interp_weight*global_weight
        return final_score


    def _compute_wups(self, ground_truths, predictions, threshold=0.925):
        scores = []
        for gt, pred in zip(ground_truths, predictions):
            # Tokenize the inputs if they are not already single words
            if isinstance(gt, list):
                gt = ' '.join(gt)
            if isinstance(pred, list):
                pred = ' '.join(pred)

            score = self._wup_measure(gt, pred, threshold)
            scores.append(score)
        return sum(scores) / len(scores)

    def batch_wup_measure(self, inverse_labels, labels, preds):
        """
        Compute WUP score. 
        Parameters:
        - labels (list of str): List of ground truth labels.
        - preds (list of str): List of predicted labels.

        Returns:
        - list of float: List of WUP scores.

        """
        try:
            logger.log_message("info", "Computing WUP score...")
            wup_scores = []
            lis_new_str = [inverse_labels.get(pred) for label, pred in zip(labels, preds)]
            lis_label = [inverse_labels.get(label) for label, pred in zip(labels, preds)]
            wup_scores = self._compute_wups(lis_new_str, lis_label)
            return wup_scores

        except Exception as e:
            print(MyException("Error computing WUP score", e))

    
    def _accuracy_score_func(self, labels, preds):
        """
        Calculate accuracy score.

        Parameters:
        - labels (list of str): List of ground truth labels.
        - preds (list of str): List of predicted labels.

        Returns:
        - float: The accuracy score.
        """
        try:
            logger.log_message("info", "Computing accuracy score...")
            return accuracy_score(self, labels, preds)

        except Exception as e:
            print(MyException("Error computing accuracy score", e))

    def _f1_score_func(self, labels, preds):
        """
        Calculate F1 score.

        Parameters:
        - labels (list of str): List of ground truth labels.
        - preds (list of str): List of predicted labels.

        Returns:
        - float: The F1 score.
        """
        try:
            logger.log_message("info", "Computing F1 score...")

            return f1_score(labels, preds, average='macro')

        except Exception as e:
            print(MyException("Error computing F1 score", e))

    ### MRR metrics
    def _mean_reciprocal_rank(self, ground_truth, predictions):
        """
        Calculate Mean Reciprocal Rank (MRR).

        Parameters:
        - predictions (list of lists of str): List of lists of predicted answers ranked by confidence.
        - ground_truth (list of str): List of ground truth answers.

        Returns:
        - float: The MRR score.
        """

        logger.log_message("info", "Computing MRR score...")
        rr_sum = 0.0
        for i, gt in enumerate(ground_truth):
            try:
                rank = predictions[i].index(gt) + 1
                # print(rank)
                rr_sum += 1.0 / rank
                # print(1.0 / rank)
            except ValueError:
                continue  # Ground truth not found in predictions

        # print(len(ground_truth))
        mrr = rr_sum / len(ground_truth)
        return mrr

    def compute_metrics(self, convert_text_answer: Dict[int, str],
                 eval_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
        try:
            logger.log_message("info", "Computing metrics...")
            preds, preds_top, labels  = eval_tuple
            # preds = logits.argmax(axis=-1)
            return {
                "wups": self._batch_wup_measure(convert_text_answer, labels, preds),
                # "blue": batch_bleu_measure(convert_text_answer, labels, preds),
                # "meteor": batch_meteor_measure(convert_text_answer, labels, preds),
                "acc": self._accuracy_score_func(labels, preds),
                "f1": self._f1_score_func(labels, preds),
                "mrr": self._mean_reciprocal_rank(labels, preds_top)
            }

        except Exception as e:
            print(MyException("Error computing metrics", e))
