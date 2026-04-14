"""
Test the advanced CF model
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io import load_train
from src.metrics import mrr_at_k, accuracy
from src.models.cf_advanced import AdvancedUserUserCF


def validate_model(model, x_va, y_va, theta_range=None, desc=""):
    """Validate a model and find best theta"""
    if theta_range is None:
        theta_range = np.linspace(0.0, 1.0, 51)
    
    # Generate predictions
    pred_jobs = []
    pred_scores = []
    for jobs, actions in tqdm(zip(x_va["job_ids"], x_va["actions"]), 
                              total=len(x_va), desc=f"Predicting {desc}"):
        top10, scores10 = model.predict_top10_with_scores(list(jobs), list(actions))
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
    
    # Split train/val
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x_train))
    cut = int(0.8 * len(x_train))
    
    x_tr = x_train.iloc[perm[:cut]].copy()
    x_va = x_train.iloc[perm[cut:]].copy()
    
    y_tr = x_tr[["session_id"]].merge(y_train, on="session_id", how="left")
    y_va = x_va[["session_id"]].merge(y_train, on="session_id", how="left")
    
    print("=" * 80)
    print("TESTING ADVANCED CF WITH VARIANTS")
    print("=" * 80)
    
    results = {}
    
    # Test different configurations
    configs = [
        ("Standard", False, False),
        ("Recency Only", True, False),
        ("Action Only", False, True),
        ("Recency + Action", True, True),
    ]
    
    for name, recency, action_w in configs:
        print(f"\n[{name}] (recency={recency}, action_weight={action_w})")
        
        model = AdvancedUserUserCF(
            k_neighbors=50,
            recency_weight=recency,
            action_weight=action_w
        )
        model.fit(x_tr, y_tr)
        
        res = validate_model(model, x_va, y_va, desc=name)
        results[name] = (model, res)
        
        print(f"  MRR:      {res['mrr']:.4f}")
        print(f"  Best ACC: {res['best_acc']:.4f}")
        print(f"  Best θ:   {res['best_theta']:.3f}")
        print(f"  FINAL:    {res['final_score']:.4f}")
    
    # Find best configuration
    print("\n" + "=" * 80)
    best_name = max(results.keys(), key=lambda k: results[k][1]['final_score'])
    best_model, best_res = results[best_name]
    
    print(f"BEST CONFIG: {best_name}")
    print(f"  MRR:      {best_res['mrr']:.4f}")
    print(f"  Best ACC: {best_res['best_acc']:.4f}")
    print(f"  Best θ:   {best_res['best_theta']:.3f}")
    print(f"  FINAL:    {best_res['final_score']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
