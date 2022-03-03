import importlib
import sys
import time
import os
import pandas as pd
from datetime import datetime
# Benchmark performance of given scripts

# Here's how it works:
#     First argument is whether we are benchmarking a preprocessing step or a model training step
#     Second argument is the script we are running

def main():
    if len(sys.argv) != 2:
        print("ERROR: Script must receive two arguments: the first indicating whether it is for a preprocessing script or a model training script, and the second for the script itself")
        sys.exit(0)
    #IMPORT SCRIPT MODULE
    script = sys.argv[1]
    mod = importlib.import_module(script)
    #START TIME
    start = time.perf_counter()
    #RUN SCRIPT
    result = mod.main()
    #STOP TIME
    end = time.perf_counter()
    #CALCULATE ELAPSED
    elapsed = end-start
    #GET DATETIME
    dt = datetime.now()
    if sys.argv == "preprocess":
        update_benchmark_preprocess(script, dt, elapsed)
    elif sys.argv == "model":
        update_benchmark_model(script, dt, elapsed, result)

def update_benchmark_model(script, dt, elapsed, result):
    # script | datetime of trial | time elapsed | final_train_accuracy | best train_accuracy | final test_accuracy | best test_accuracy
    if os.path.getsize("benchmark_model.csv") == 0:
        df = pd.DataFrame(columns = ["script", "datetime", "time_elapsed", "final_train_accuracy", "best_train_accuracy", "final_test_accuracy", "best_test_accuracy"])
    else:
        df = pd.read_csv('benchmark_model.csv')
    #Add row to dataframe
    row = {
        "script": script,
        "datetime": dt,
        "time_elapsed": elapsed,
        "final_train_accuracy": result["final_train_accuracy"],
        "best_train_accuracy": result["best_train_accuracy"],
        "final_test_accuracy": result["best_test_accuracy"],
        "best_test_accuracy": result["best_test_accuracy"]
    }
    df.append(row, ignore_index=True)
    df.to_csv("benchmark_model.csv")
    return

def update_benchmark_preprocess(script, dt, elapsed):
    # script | datetime of trial | time elapsed
    if os.path.getsize("benchmark_preprocess.csv") == 0:
        df = pd.DataFrame(columns = ["script", "datetime", "time_elapsed"])
    else:
        df = pd.read_csv('benchmark_preprocess.csv')
    #Add row to dataframe
    row = {
        "script": script,
        "datetime": dt,
        "time_elapsed": elapsed
    }
    df.append(row, ignore_index=True)
    df.to_csv("benchmark_preprocess.csv")
    return