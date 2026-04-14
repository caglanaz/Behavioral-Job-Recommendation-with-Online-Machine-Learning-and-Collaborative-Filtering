"""
Ensemble model combining CF, Markov, and contextual patterns.
"""
import numpy as np
from collections import Counter, defaultdict


class ContextualPatternModel:
    """Capture local patterns from recent jobs"""
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.patterns = defaultdict(Counter)  # tuple of (job1, job2, ...) -> Counter of next jobs
        self.global_pop = []
    
    def fit(self, x_train, y_train):
        pop_counts = Counter(y_train["job_id"])
        self.global_pop = [jid for jid, _ in pop_counts.most_common(300)]
        
        # Build patterns from recent jobs
        for jobs in x_train["job_ids"]:
            jobs = list(jobs)
            for i in range(len(jobs)):
                # Get last `window_size` jobs before position i
                start = max(0, i - self.window_size + 1)
                pattern = tuple(jobs[start:i+1])
                
                # Try to find next job in y_train
                # For now, just count patterns locally
                if len(jobs) > i + 1:
                    next_job = jobs[i + 1]
                    self.patterns[pattern][next_job] += 1
    
    def predict_top10_with_scores(self, job_ids, y_train=None):
        """Predict next 10 jobs based on contextual patterns"""
        job_ids = list(job_ids)
        cand = []
        scores = []
        
        # Try patterns of increasing specificity
        for window in range(min(self.window_size, len(job_ids)), 0, -1):
            pattern = tuple(job_ids[-window:])
            if pattern in self.patterns and self.patterns[pattern]:
                total = sum(self.patterns[pattern].values())
                for jid, count in self.patterns[pattern].most_common(10):
                    prob = count / total if total > 0 else 0.0
                    cand.append(jid)
                    scores.append(prob)
        
        # Fallback to global popularity
        seen = set(cand)
        for jid in self.global_pop:
            if jid not in seen:
                cand.append(jid)
                scores.append(0.0)
                seen.add(jid)
            if len(cand) == 10:
                break
        
        return cand[:10], scores[:10]


class EnsembleRecommender:
    """Ensemble combining CF, Markov, and Contextual models"""
    
    def __init__(self, cf_model, markov_model, contextual_model=None, 
                 cf_weight=0.5, markov_weight=0.3, contextual_weight=0.2):
        self.cf = cf_model
        self.markov = markov_model
        self.contextual = contextual_model
        self.cf_weight = cf_weight
        self.markov_weight = markov_weight
        self.contextual_weight = contextual_weight
    
    def predict_top10_with_scores(self, job_ids):
        """Combine predictions from multiple models"""
        # Get predictions from each model
        cf_jobs, cf_scores = self.cf.predict_top10_with_scores(job_ids)
        markov_jobs, markov_scores = self.markov.predict_top10_with_scores(job_ids)
        
        # Initialize scores dictionary
        combined_scores = defaultdict(float)
        
        # Add CF scores
        for job, score in zip(cf_jobs, cf_scores):
            combined_scores[job] += self.cf_weight * (score + 1.0)  # Boost with offset
        
        # Add Markov scores
        for job, score in zip(markov_jobs, markov_scores):
            combined_scores[job] += self.markov_weight * (score + 0.5)
        
        # Add contextual scores if available
        if self.contextual:
            ctx_jobs, ctx_scores = self.contextual.predict_top10_with_scores(job_ids)
            for job, score in zip(ctx_jobs, ctx_scores):
                combined_scores[job] += self.contextual_weight * (score + 0.3)
        
        # Normalize scores
        seen = set(job_ids)
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_jobs, top_scores = [], []
        for job, score in ranked:
            if job not in seen:
                top_jobs.append(job)
                top_scores.append(score)
                if len(top_jobs) == 10:
                    break
        
        # Add fallback from CF if needed
        if len(top_jobs) < 10:
            for job in cf_jobs:
                if job not in seen and job not in top_jobs:
                    top_jobs.append(job)
                    top_scores.append(0.0)
                    if len(top_jobs) == 10:
                        break
        
        return top_jobs[:10], top_scores[:10]
    
    def predict_apply_from_scores(self, scores, theta):
        """Predict action based on score threshold"""
        p_mean = float(np.mean(scores)) if scores else 0.0
        return "apply" if p_mean >= theta else "view"
