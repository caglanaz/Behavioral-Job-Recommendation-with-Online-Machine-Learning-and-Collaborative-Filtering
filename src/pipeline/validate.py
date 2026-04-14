from src.io import load_train
from src.metrics import mrr_at_k, accuracy
from src.models.cf_user_user import UserUserCF
import numpy as np

x_train, y_train = load_train(
    "data/raw/x_train_Meacfjr.csv",
    "data/raw/y_train_SwJNMSu.csv"
)

rng = np.random.default_rng(42)
perm = rng.permutation(len(x_train))
cut = int(0.8 * len(x_train))

x_tr = x_train.iloc[perm[:cut]].copy()
x_va = x_train.iloc[perm[cut:]].copy()

y_tr = x_tr[["session_id"]].merge(y_train, on="session_id", how="left")
y_va = x_va[["session_id"]].merge(y_train, on="session_id", how="left")

model = UserUserCF(k_neighbors=50)
model.fit(x_tr)

pred_jobs = []
pred_scores = []
for jobs in x_va["job_ids"]:
    top10, scores10 = model.predict_top10_with_scores(jobs)
    pred_jobs.append(top10)
    pred_scores.append(scores10)

# --- chercher theta sur validation
thetas = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00
best = (-1, None, None, None)

for theta in thetas:
    pred_act = [model.predict_apply_from_scores(s, theta) for s in pred_scores]
    mrr = mrr_at_k(y_va["job_id"].tolist(), pred_jobs, k=10)
    acc = accuracy(y_va["action"].tolist(), pred_act)
    final = 0.7*mrr + 0.3*acc
    if final > best[0]:
        best = (final, theta, mrr, acc)

final, theta, mrr, acc = best
print("Best theta:", theta)
print("MRR:", mrr)
print("ACC:", acc)
print("Final:", final)
