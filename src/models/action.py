def predict_action_simple(actions):
    return "apply" if any(a == "apply" for a in actions) else "view"
