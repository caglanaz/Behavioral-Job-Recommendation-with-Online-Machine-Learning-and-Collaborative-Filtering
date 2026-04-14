"""
Advanced CF model with recency weighting and action-aware similarity
"""
import numpy as np
from collections import Counter, defaultdict


class AdvancedUserUserCF:
    """
    Enhanced User-User CF with:
    - Recency weighting (recent jobs matter more)
    - Action-aware similarity (applies vs views)
    - Better similarity metric
    """
    
    def __init__(self, k_neighbors=50, recency_weight=True, action_weight=True):
        self.k = k_neighbors
        self.recency_weight = recency_weight
        self.action_weight = action_weight
        self.train_sessions = []
        self.train_actions = []
        self.global_pop = []
        self.global_apply_rate = {}
    
    def fit(self, x_train, y_train):
        """Fit the model with weighted job preferences"""
        self.train_sessions = [list(seq) for seq in x_train["job_ids"]]
        self.train_actions = [list(seq) for seq in x_train["actions"]]
        
        # Compute global popularity and apply rates
        pop = Counter()
        apply_counts = Counter()
        total_counts = Counter()
        
        for jobs, actions in zip(self.train_sessions, self.train_actions):
            for job, action in zip(jobs, actions):
                pop[job] += 1
                total_counts[job] += 1
                if action == "apply":
                    apply_counts[job] += 1
        
        self.global_pop = [j for j, _ in pop.most_common(500)]
        
        # Compute apply rate for each job
        for job in total_counts:
            self.global_apply_rate[job] = apply_counts[job] / total_counts[job]
    
    def _create_weighted_profile(self, job_ids, actions):
        """Create a weighted profile vector from job sequence"""
        profile = defaultdict(float)
        
        if not job_ids:
            return profile
        
        n = len(job_ids)
        for i, (job, action) in enumerate(zip(job_ids, actions)):
            # Recency weight: exponential decay from recent to old
            if self.recency_weight:
                pos_weight = (i + 1) / n  # Linear from 0 to 1
            else:
                pos_weight = 1.0
            
            # Action weight: apply is stronger signal than view
            if self.action_weight:
                action_weight = 2.0 if action == "apply" else 1.0
            else:
                action_weight = 1.0
            
            profile[job] += pos_weight * action_weight
        
        # Normalize
        total = sum(profile.values())
        if total > 0:
            for job in profile:
                profile[job] /= total
        
        return profile
    
    def _similarity_weighted_jaccard(self, profile_a, profile_b):
        """Compute similarity between two weighted profiles"""
        if not profile_a or not profile_b:
            return 0.0
        
        jobs_a = set(profile_a.keys())
        jobs_b = set(profile_b.keys())
        
        if not jobs_a or not jobs_b:
            return 0.0
        
        # Intersection: sum of min weights
        intersection = 0.0
        for job in jobs_a & jobs_b:
            intersection += min(profile_a[job], profile_b[job])
        
        # Union: sum of max weights
        union = 0.0
        for job in jobs_a | jobs_b:
            w_a = profile_a.get(job, 0.0)
            w_b = profile_b.get(job, 0.0)
            union += max(w_a, w_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def predict_top10_with_scores(self, job_ids, actions=None):
        """Predict top 10 next jobs"""
        # Default to all views if no actions provided
        if actions is None:
            actions = ["view"] * len(job_ids)
        
        # Create profile for this session
        query_profile = self._create_weighted_profile(job_ids, actions)
        
        # Find similar sessions
        similarities = []
        for idx, (session_jobs, session_actions) in enumerate(
            zip(self.train_sessions, self.train_actions)):
            session_profile = self._create_weighted_profile(session_jobs, session_actions)
            sim = self._similarity_weighted_jaccard(query_profile, session_profile)
            if sim > 0:
                similarities.append((sim, idx))
        
        # Keep top-k neighbors
        similarities.sort(reverse=True)
        similarities = similarities[:self.k]
        
        # Score candidates by weighted frequency in neighbor sessions
        candidate_scores = defaultdict(float)
        if similarities:
            total_sim = sum(sim for sim, _ in similarities)
            for sim, idx in similarities:
                weight = sim / total_sim
                session_jobs = self.train_sessions[idx]
                session_actions = self.train_actions[idx]
                
                # Weight each job by its strength in the neighbor session
                for job, action in zip(session_jobs, session_actions):
                    action_mult = 2.0 if action == "apply" else 1.0
                    candidate_scores[job] += weight * action_mult
        
        # Rank and return top 10
        seen = set(job_ids)
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_jobs, top_scores = [], []
        for job, score in ranked:
            if job not in seen:
                top_jobs.append(job)
                top_scores.append(score)
                if len(top_jobs) == 10:
                    break
        
        # Fallback to global popularity
        for job in self.global_pop:
            if job not in seen and job not in top_jobs:
                top_jobs.append(job)
                top_scores.append(0.0)
                if len(top_jobs) == 10:
                    break
        
        return top_jobs[:10], top_scores[:10]
    
    def predict_apply_from_scores(self, scores, theta):
        """Predict action based on score threshold"""
        if not scores:
            return "view"
        p_mean = float(np.mean(scores))
        return "apply" if p_mean >= theta else "view"
