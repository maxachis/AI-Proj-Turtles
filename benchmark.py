import sys, time
# Benchmark performance of given scripts

# Here's how it works:
#     First argument is whether we are benchmarking a preprocessing step or a model training step
#     Second argument is the script we are running

if len(sys.argv) != 2:
    print("ERROR: Script must receive two arguments: the first indicating whether it is for a preprocessing script or a model training script, and the second for the script itself")
    sys.exit(0)

start = time.perf_counter()



end = time.perf_counter()



def update_benchmark_model():
    # script | datetime of trial | time elapsed | final train_accuracy | best train_accuracy | final test_accuracy | best test_accuracy
    return

def update_benchmark_preprocess():
    # script | datetime of trial | time elapsed
    return
