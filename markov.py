from collections import Counter, defaultdict
import numpy as np

class MarkovModel:
    def __init__(self):
        self.trans = defaultdict(Counter)
        self.top_pop = []
        self.last_job_map = {}

    def fit(self, x_train, y_train):
        pop_counts = Counter(y_train["job_id"])
        self.top_pop = [jid for jid, _ in pop_counts.most_common(200)]

        self.last_job_map = dict(zip(
            x_train["session_id"],
            x_train["job_ids"].apply(lambda l: l[-1])
        ))

        for sid, target in zip(y_train["session_id"], y_train["job_id"]):
            lj = self.last_job_map.get(sid)
            if lj is not None:
                self.trans[lj][target] += 1

    def predict_top10_with_scores(self, job_ids):
        """Retourne top 10 jobs avec scores (probabilités de transition)"""
        lj = job_ids[-1]
        cand = []
        scores = []

        if lj in self.trans:
            total_count = sum(self.trans[lj].values())
            for jid, count in self.trans[lj].most_common(10):
                prob = count / total_count if total_count > 0 else 0.0
                cand.append(jid)
                scores.append(prob)

        # Fallback sur popularité globale avec score 0
        seen = set(cand)
        for jid in self.top_pop:
            if jid not in seen:
                cand.append(jid)
                scores.append(0.0)
                seen.add(jid)
            if len(cand) == 10:
                break

        return cand[:10], scores[:10]
    
    def predict_apply_from_scores(self, scores, theta):
        """Predict action based on score threshold"""
        p_mean = float(np.mean(scores)) if scores else 0.0
        return "apply" if p_mean >= theta else "view"
