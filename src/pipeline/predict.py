import numpy as np
import pandas as pd

from src.io import load_train, load_test
from src.metrics import mrr_at_k, accuracy
from src.models.cf_user_user import UserUserCF


def find_best_theta(x_train, y_train, k_neighbors=50, seed=42):
    """
    Cherche theta sur un split interne du train (train/valid).
    Retourne le meilleur theta selon le score final 0.7*MRR + 0.3*ACC.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_train))
    cut = int(0.8 * len(x_train))

    x_tr = x_train.iloc[perm[:cut]].copy()
    x_va = x_train.iloc[perm[cut:]].copy()

    # aligner y sur l'ordre de x
    y_tr = x_tr[["session_id"]].merge(y_train, on="session_id", how="left")
    y_va = x_va[["session_id"]].merge(y_train, on="session_id", how="left")

    model = UserUserCF(k_neighbors=k_neighbors)
    model.fit(x_tr)

    pred_jobs, pred_scores = [], []
    for jobs in x_va["job_ids"]:
        top10, scores10 = model.predict_top10_with_scores(jobs)
        pred_jobs.append(top10)
        pred_scores.append(scores10)

    thetas = np.linspace(0.0, 1.0, 51)  # 0.00 -> 1.00 step 0.02
    best_final, best_theta, best_mrr, best_acc = -1.0, 0.0, 0.0, 0.0

    y_job = y_va["job_id"].tolist()
    y_act = y_va["action"].tolist()

    mrr = mrr_at_k(y_job, pred_jobs, k=10)  # ne dépend pas de theta
    for theta in thetas:
        pred_act = [model.predict_apply_from_scores(s, theta) for s in pred_scores]
        acc = accuracy(y_act, pred_act)
        final = 0.7 * mrr + 0.3 * acc
        if final > best_final:
            best_final, best_theta, best_mrr, best_acc = final, float(theta), float(mrr), float(acc)

    return best_theta, best_mrr, best_acc, best_final


def main():
    # --- paths (adapte si tu renommes)
    x_train_path = "data/raw/x_train_Meacfjr.csv"
    y_train_path = "data/raw/y_train_SwJNMSu.csv"
    x_test_path  = "data/raw/x_test_jCBBNP2.csv"

    x_train, y_train = load_train(x_train_path, y_train_path)
    x_test = load_test(x_test_path)

    # --- hyperparams simples
    k_neighbors = 50

    # 1) calibrer theta sur split interne
    theta, mrr, acc, final = find_best_theta(x_train, y_train, k_neighbors=k_neighbors, seed=42)
    print(f"[Calibration] k={k_neighbors} theta={theta:.3f} | MRR={mrr:.4f} ACC={acc:.4f} FINAL={final:.4f}")

    # 2) refit sur tout le train
    model = UserUserCF(k_neighbors=k_neighbors)
    model.fit(x_train)

    # 3) prédire sur test
    out_session = []
    out_action = []
    out_jobs = []

    for sid, jobs in zip(x_test["session_id"], x_test["job_ids"]):
        top10, scores10 = model.predict_top10_with_scores(jobs)
        act = model.predict_apply_from_scores(scores10, theta)

        out_session.append(int(sid))
        out_action.append(act)
        out_jobs.append(str(top10))  # IMPORTANT: format "[..., ...]"

    submission = pd.DataFrame({
        "session_id": out_session,
        "action": out_action,
        "job_id": out_jobs
    })

    # 4) sauver
    out_path = "outputs/submissions/submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
