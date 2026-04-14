"""
Advanced validation script with multiple models and better tuning
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io import load_train, load_test
from src.metrics import mrr_at_k, accuracy
from src.models.cf_user_user import UserUserCF
from src.models.markov import MarkovModel
from src.models.ensemble import EnsembleRecommender, ContextualPatternModel


def validate_model(model, x_va, y_va, theta_range=None):
    """Validate a model and find best theta"""
    if theta_range is None:
        theta_range = np.linspace(0.0, 1.0, 51)
    
    # Generate predictions
    pred_jobs = []
    pred_scores = []
    for jobs in tqdm(x_va["job_ids"], desc="Predicting"):
        top10, scores10 = model.predict_top10_with_scores(jobs)
        pred_jobs.append(top10)
        pred_scores.append(scores10)
    
    # Compute MRR (doesn't depend on theta)
    y_job = y_va["job_id"].tolist()
    mrr = mrr_at_k(y_job, pred_jobs, k=10)
    
    # Find best theta
    y_act = y_va["action"].tolist()
    best_final, best_theta, best_acc = -1.0, 0.0, 0.0
    
    for theta in theta_range:
        pred_act = [model.predict_apply_from_scores(s, theta) for s in pred_scores]
        acc = accuracy(y_act, pred_act)
        final = 0.7 * mrr + 0.3 * acc
        if final > best_final:
            best_final = final
            best_theta = float(theta)
            best_acc = float(acc)
    
    return {
        'mrr': float(mrr),
        'best_acc': best_acc,
        'best_theta': best_theta,
        'final_score': best_final,
        'pred_jobs': pred_jobs,
        'pred_scores': pred_scores
    }


def main():
    # Load data
    x_train, y_train = load_train(
        "data/raw/x_train_Meacfjr.csv",
        "data/raw/y_train_SwJNMSu.csv"
    )
    x_test = load_test("data/raw/x_test_jCBBNP2.csv")
    
    # Split train/val
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    
    x_tr = x_train.iloc[perm[:cut]].copy()
    x_va = x_train.iloc[perm[cut:]].copy()
    
    y_tr = x_tr[["session_id"]].merge(y_train, on="session_id", how="left")
    y_va = x_va[["session_id"]].merge(y_train, on="session_id", how="left")
    
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # Test CF with different k values
    for k in [30, 50, 70, 100]:
        print(f"\n[CF] k={k}")
        cf = UserUserCF(k_neighbors=k)
        cf.fit(x_tr)
        
        res = validate_model(cf, x_va, y_va)
        model_name = f"CF_k{k}"
        results[model_name] = (cf, res)
        
        print(f"  MRR:      {res['mrr']:.4f}")
        print(f"  Best ACC: {res['best_acc']:.4f}")
        print(f"  Best θ:   {res['best_theta']:.3f}")
        print(f"  FINAL:    {res['final_score']:.4f}")
    
    # Test Markov
    print(f"\n[Markov]")
    markov = MarkovModel()
    markov.fit(x_tr, y_tr)
    
    res = validate_model(markov, x_va, y_va)
    results["Markov"] = (markov, res)
    
    print(f"  MRR:      {res['mrr']:.4f}")
    print(f"  Best ACC: {res['best_acc']:.4f}")
    print(f"  Best θ:   {res['best_theta']:.3f}")
    print(f"  FINAL:    {res['final_score']:.4f}")
    
    # Test Ensemble
    print(f"\n[Ensemble CF+Markov]")
    cf_best = UserUserCF(k_neighbors=50)
    cf_best.fit(x_tr)
    markov_best = MarkovModel()
    markov_best.fit(x_tr, y_tr)
    
    ensemble = EnsembleRecommender(cf_best, markov_best, 
                                   cf_weight=0.6, markov_weight=0.4)
    
    res = validate_model(ensemble, x_va, y_va)
    results["Ensemble"] = (ensemble, res)
    
    print(f"  MRR:      {res['mrr']:.4f}")
    print(f"  Best ACC: {res['best_acc']:.4f}")
    print(f"  Best θ:   {res['best_theta']:.3f}")
    print(f"  FINAL:    {res['final_score']:.4f}")
    
    # Find best model
    print("\n" + "=" * 80)
    best_model_name = max(results.keys(), key=lambda k: results[k][1]['final_score'])
    best_model, best_res = results[best_model_name]
    
    print(f"BEST MODEL: {best_model_name}")
    print(f"FINAL SCORE: {best_res['final_score']:.4f}")
    print("=" * 80)
    
    return best_model, best_res['best_theta']


if __name__ == "__main__":
    best_model, best_theta = main()
