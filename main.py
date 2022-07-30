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


        itemindex = sorted(list(set(y_true).union(y_pred)))
        t_vecs, p_vecs = [], []
        for t, p in zip(y_true, y_pred):
                t_vec = [1 if itemindex[i]==t else 0 for i in range(len(itemindex))]
                p_vec = [1 if itemindex[i]==p else 0 for i in range(len(itemindex))]
                t_vecs.append(t_vec)
                p_vecs.append(p_vec)

        y_true, y_pred = t_vecs, p_vecs

        tc = TruthConditionalMetric(itemindex=itemindex)
        tc.build(y_trues=y_true, y_preds=y_pred, verbose=True)
        # tc.get_metric(verbose=True)

        pass
