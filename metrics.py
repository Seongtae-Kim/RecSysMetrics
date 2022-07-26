from functools import reduce
from math_functions import *

# Seongtae Kim / 2022-07-26
# Reference: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826


class RecSysMetric:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("Abstract Implementation Error")
    
    def assert_lengths_equal(self, y_true, y_pred):
        try:
            assert len(y_pred) == len(y_true)
        except AssertionError:
            raise Exception("the length of y_pred and y_true must be the same.")

    def assert_k(self, y_true, y_pred, k):
        try:
            assert k <= min(len(y_pred), len(y_true))
        except AssertionError:
            raise Exception(f"the length of k must be equal or small than the the smallest length between y_pred and y_true (min_length={min(len(y_pred), len(y_true))}).")


class RMSE(RecSysMetric):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        self.assert_lengths_equal(y_true, y_pred)
        N = len(y_true)

        def calculate_rmse(a, b): # For custom function for reduce
            if a is None and b is not None: return float(b)
            if b is None and a is not None: return float(a)
            if not isinstance(a, tuple) and not isinstance(b, tuple): return float(a + b)
            if isinstance(a, tuple) and not isinstance(b, tuple): return float(((a[1]-a[0])**2) + b )
            if not isinstance(a, tuple) and isinstance(b, tuple): return float(a + ((b[1]-b[0])**2))
            if isinstance(a, tuple) and isinstance(b, tuple): return float(((a[1]-a[0])**2) + ((b[1]-b[0])**2))

        return sqrt(reduce(calculate_rmse, list(zip(y_true, y_pred)))/N)

class NDCG(RecSysMetric):
    def __init__(self):
        super().__init__()
    def __call__(self, y_true, y_pred):
        self.assert_lengths_equal(y_true, y_pred)
        dcg = lambda y_seq : sum([y_seq[i-1]/log2(i+1) for i in range(1, len(y_seq)+1)])
        return dcg(y_pred)/dcg(y_true)


class TruthConditionalMetric(RecSysMetric):
    def __init__(self):
        super().__init__()
        self.confusion_matrix = None

    def build(self, y_trues, y_preds, k=None, verbose=False):
        self.assert_lengths_equal(y_trues, y_preds)
        
        self.confusion_matrix={}        
        for y_true, y_pred in zip(y_trues, y_preds):
            self.assert_lengths_equal(y_true, y_pred)
            
            if k: # For Precision@K
                self.assert_k(y_true, y_pred, k)
                y_true, y_pred = y_true[:k], y_pred[:k]

            for t, p in zip(y_true, y_pred):
                # Actual
                self.confusion_matrix.setdefault(t, {t:0, p:0})
                self.confusion_matrix.setdefault(p, {p:0, t:0})

                # Predicted
                self.confusion_matrix[t][p] = self.confusion_matrix[t].get(p, 0) + 1

        if verbose:
            for t in self.confusion_matrix:
                for p in self.confusion_matrix[t]:
                    print(f"<TRUE={t}> <PRED={p}>", end=" ")
                    v = self.confusion_matrix[t][p]
                    print(v)


    def get_metric(self, verbose=False):
        if not self.confusion_matrix:
            raise Exception("You Must Build Confusion Matrix First!")

        metric={}
        for true in self.confusion_matrix:
            metric.setdefault(true, {})

            # TRUE-POSITIVE
            metric[true]["TP"] = self.confusion_matrix[true][true] 

            # TRUE-NEGATIVE
            metric[true]["TN"] = sum([sum([self.confusion_matrix[actual][pred] for pred in self.confusion_matrix[actual] if pred != true]) for actual in self.confusion_matrix if actual != true])

            # FALSE-NEGATIVE
            metric[true]["FN"] = sum([sum([self.confusion_matrix[actual][pred] for pred in self.confusion_matrix[actual] if pred == true]) for actual in self.confusion_matrix if actual != true]) 

            # FALSE-POSITIVE
            metric[true]["FP"] = sum([sum([self.confusion_matrix[actual][pred] for pred in self.confusion_matrix[actual] if pred != true]) for actual in self.confusion_matrix if actual == true]) 
            
            metric[true]["Precision"] = metric[true]["TP"] / (metric[true]["TP"] + metric[true]["FP"])
            metric[true]["Recall"] = metric[true]["TP"] / (metric[true]["TP"] + metric[true]["FN"])
            metric[true]["FalsePositiveRate"] = metric[true]["FP"] / (metric[true]["FP"] + metric[true]["TN"])
            metric[true]["F1-Score"] = 2 * metric[true]["Precision"] * metric[true]["Recall"] / (metric[true]["Precision"] + metric[true]["Recall"])

        total_meta_name = "TOTAL_RATE"
        metric[total_meta_name] = {}
        for name in ["TP", "TN", "FN", "FP", "Precision", "Recall", "FalsePositiveRate"]:
            metric[total_meta_name]["Average " + name] = sum([metric[cat_name][name] for cat_name in metric if cat_name != total_meta_name]) / (len(metric)-1)
        
        self.metric = metric
        
        # Micro F1
        micro_tp = sum([metric[cat_name]["TP"] for cat_name in metric if cat_name != total_meta_name])
        micro_fp = sum([metric[cat_name]["FP"] for cat_name in metric if cat_name != total_meta_name])
        micro_fn = sum([metric[cat_name]["FN"] for cat_name in metric if cat_name != total_meta_name])
        micro_precision = micro_tp / (micro_tp + micro_fp)
        micro_recall = micro_tp / (micro_tp + micro_fn)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        metric[total_meta_name]["Micro F1"] = micro_f1

        # Macro F1
        metric[total_meta_name]["Macro F1"] = sum([metric[cat_name]["F1-Score"] for cat_name in metric if cat_name != total_meta_name]) / (len(metric)-1)
        
        # Weighted F1
        total_numbers = {true:sum(self.confusion_matrix[true].values()) for true in self.confusion_matrix}
        metric[total_meta_name]["Weighted F1"] = sum([metric[cat_name]["F1-Score"] * total_numbers[cat_name] for cat_name in metric if cat_name != total_meta_name]) / sum(total_numbers.values())

        if verbose:
            for true in metric:
                print(f"=={true}==")
                for confusion in metric[true]:
                    print(f"\t{confusion}: {metric[true][confusion]}")

        self.metric = metric
        self.total_meta_name = total_meta_name

