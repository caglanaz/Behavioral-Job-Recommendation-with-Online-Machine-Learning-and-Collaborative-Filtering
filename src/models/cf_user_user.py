import numpy as np
from collections import Counter, defaultdict

class UserUserCF:
    def __init__(self, k_neighbors=50):
        self.k = k_neighbors
        self.train_sessions = []
        self.global_pop = []

    def fit(self, x_train):
        self.train_sessions = [set(seq) for seq in x_train["job_ids"]]

        pop = Counter()
        for s in self.train_sessions:
            pop.update(s)
        self.global_pop = [j for j, _ in pop.most_common(500)]

    @staticmethod
    def cosine_set(a, b):
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return inter / (np.sqrt(len(a)) * np.sqrt(len(b)))

    def predict_top10_with_scores(self, job_ids):
        q = set(job_ids)
        sims = []

        for idx, s in enumerate(self.train_sessions):
            sim = self.cosine_set(q, s)
            if sim > 0:
                sims.append((sim, idx))

        sims.sort(reverse=True)
        sims = sims[: self.k]

        score = defaultdict(float)
        if sims:
            # Version améliorée: pondérer par la similarité réelle
            total_sim = sum(sim for sim, _ in sims)
            if total_sim > 0:
                for sim, idx in sims:
                    weight = sim / total_sim  # Normaliser par la somme des similarités
                    for j in self.train_sessions[idx]:
                        score[j] += weight
            
            ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
            ranked_jobs = [j for j, sc in ranked]
        else:
            ranked_jobs = []

        seen = set(job_ids)
        top_jobs, top_scores = [], []

        # ranked (score CF)
        for j in ranked_jobs:
            if j in seen:
                continue
            top_jobs.append(j)
            top_scores.append(score[j])
            if len(top_jobs) == 10:
                return top_jobs, top_scores

        # fallback popularité (score = 0)
        for j in self.global_pop:
            if j in seen:
                continue
            top_jobs.append(j)
            top_scores.append(0.0)
            if len(top_jobs) == 10:
                break

        return top_jobs, top_scores

    def predict_apply_from_scores(self, top10_scores, theta):
        p_mean = float(np.mean(top10_scores)) if top10_scores else 0.0
        return "apply" if p_mean >= theta else "view"
