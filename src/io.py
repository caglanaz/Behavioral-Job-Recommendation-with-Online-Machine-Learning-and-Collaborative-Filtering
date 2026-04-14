import pandas as pd
import ast

def parse_pylist(s):
    return ast.literal_eval(s) if isinstance(s, str) else s

def load_train(path_x, path_y):
    x = pd.read_csv(path_x)
    y = pd.read_csv(path_y)

    x["job_ids"] = x["job_ids"].apply(parse_pylist)
    x["actions"] = x["actions"].apply(parse_pylist)

    return x, y

def load_test(path):
    x = pd.read_csv(path)
    x["job_ids"] = x["job_ids"].apply(parse_pylist)
    x["actions"] = x["actions"].apply(parse_pylist)
    return x
