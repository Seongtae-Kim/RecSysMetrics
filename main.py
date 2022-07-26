from metrics import NDCG, RMSE, TruthConditionalMetric

if __name__ == "__main__":

    y_true = (["Apple"] * 7) + (["Orange"] * 2) + (["Mango"] * 1)\
             + (["Apple"] * 4)\
             + (["Orange"] * 10)\
             + (["Mango"] * 12)
        
    y_pred = (["Apple"] * 7) + (["Orange"] * 2) + (["Mango"] * 1)\
            + (["Orange"] * 1) + (["Mango"] * 3)\
            + (["Apple"] * 8) + (["Mango"] * 2)\
            + (["Apple"] * 9) + (["Orange"] * 3)
        
    tc = TruthConditionalMetric()
    tc.build(y_trues=[y_true], y_preds=[y_pred])
    tc.get_metric(verbose=True)