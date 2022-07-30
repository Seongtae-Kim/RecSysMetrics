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
    def __init__(self, itemindex):
        super().__init__()
        self.itemindex=itemindex    

    def build(self, y_trues, y_preds, binary=True, k=None, verbose=False):
        """
        y_trues:
            shape: (number of instances, number of classes)
        y_preds:
            shape: (number of instances, number of classes)
        """

        self.assert_lengths_equal(y_trues, y_preds)
        if binary and k is not None:
            raise AssertionError("You must set binary arguments as FALSE when you set k")
        
        total_metrics, class_freq = self.get_basic_metrics(y_trues, y_preds, binary)
        for class_name in total_metrics:
            base_metric = total_metrics[class_name]
            total_metrics[class_name] = self.get_advanced_metrics(base_metric)
        global_metric = self.get_global_metric(total_metrics, class_freq)


        if verbose:
            print("Stats by class")
            for class_name in total_metrics:
                print("class:",class_name)
                for stat_type in total_metrics[class_name]:
                    print(f"  {stat_type}:", total_metrics[class_name][stat_type])
                print()
            print()
            print("Global stats")
            for stat_type in global_metric:
                print(f"  {stat_type}:", global_metric[stat_type])

                    


    def get_global_metric(self, total_metrics, class_freq):
        support_sum = sum(class_freq.values())
        global_metric = {}
        for class_name in total_metrics:
            for stat_type in total_metrics[class_name]:
                if stat_type in ["TP", "TN", "FP", "FN"]:
                    global_metric[stat_type] = global_metric.get(stat_type, 0) + total_metrics[class_name][stat_type]
                elif stat_type in ["Precision", "Recall"]:
                    global_metric.setdefault("Average " + stat_type, []).append(total_metrics[class_name][stat_type])
                elif stat_type in ["F1-Score"]: # For Weighted F1 Score using support values
                    global_metric["Weighted-Average " + stat_type] = global_metric.get("Weighted-Average " + stat_type, 0.) + (total_metrics[class_name][stat_type] * class_freq[class_name] / support_sum)

        
        global_metric = self.get_advanced_metrics(global_metric)
        global_metric["Average Precision"] = sum(global_metric["Average Precision"]) / len(global_metric["Average Precision"])
        global_metric["Average Recall"] = sum(global_metric["Average Recall"]) / len(global_metric["Average Recall"])
        global_metric["Micro-Average F1 Score"] = 2 * (global_metric["Precision"] * global_metric["Recall"]) / (global_metric["Precision"] + global_metric["Recall"])
        global_metric["Macro-Average F1 Score"] = 2 * (global_metric["Average Precision"] * global_metric["Average Recall"]) / (global_metric["Average Precision"] + global_metric["Average Recall"])
        
        return global_metric


    def get_advanced_metrics(self, base_metric):
        notlisted=list(set(["TP", "TN", "FP", "FN"]).difference(set(base_metric.keys())))
        if len(notlisted) != 0:
            raise AssertionError(f"The following stats are not available: {notlisted}")

        adv_metric = {
            "Accuracy": (base_metric["TP"]+base_metric["TN"]) / (base_metric["TP"]+base_metric["TP"]+base_metric["TN"]+base_metric["FN"]),
            "Precision":(base_metric["TP"])/(base_metric["TP"]+base_metric["FP"]),
            "Recall":(base_metric["TP"])/(base_metric["TP"]+base_metric["FN"]),
            "Sensitivity": (base_metric["TP"])/(base_metric["TP"]+base_metric["FN"]),
            "Specificity": (base_metric["TN"])/(base_metric["TN"]+base_metric["FP"]),
            "False Positive Rate": (base_metric["FP"])/(base_metric["FP"]+base_metric["TN"]),
        }
        adv_metric["Misclassification Rate"] = 1 - adv_metric["Accuracy"]
        adv_metric["F1-Score"] = 2 * (adv_metric["Precision"] * adv_metric["Recall"] / (adv_metric["Precision"] + adv_metric["Recall"]))

        return adv_metric | base_metric


    def get_basic_metrics(self, y_trues, y_preds, binary=True):
        total_metrics = {item : {'TP':0, 'TN':0, 'FP':0, 'FN':0} for item in self.itemindex}
        class_freq = {item:0 for item in self.itemindex} # For Weighted F1-Score
        for y_true, y_pred in zip(y_trues, y_preds):
            self.assert_lengths_equal(y_true, y_pred)
            for i in range(len(y_true)):
                true_val, pred_val = y_true[i], y_pred[i]
                class_name = self.itemindex[i]

                if binary:
                    if true_val == 1:
                        class_freq[class_name]+=1

                    if true_val == pred_val and true_val == 1:
                        total_metrics[class_name]["TP"]+=1
                    elif true_val == pred_val and true_val == 0:
                        total_metrics[class_name]["TN"]+=1
                    elif true_val == 1 and pred_val == 0:
                        total_metrics[class_name]["FP"]+=1
                    elif true_val == 0 and pred_val == 1:
                        total_metrics[class_name]["FN"]+=1

                elif not binary:
                    pass

        return total_metrics, class_freq
