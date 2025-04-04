import concurrent.futures
import json
from statistics import mean, stdev
from local_ect import *

# Define the datasets and their respective parameters
datasets = [
    ('Actor', {'root': '/tmp/actor'}),
    ('Reddit', {'root': '/tmp/reddit'}),
    ('CS', {'root': '/tmp/CS', 'name': 'CS'}),
    ('Physics', {'root': '/tmp/Physics', 'name': 'Physics'}),
]

# All combinations of radius1 and radius2 except (False, False)
radius_combinations = [(True, True), (True, False), (False, True)]

# Number of times each configuration should be executed
num_runs = 5


def run_xgb_model_once(dataset_name, dataset_params, radius1, radius2):
    # Determine the metric based on dataset type
    if dataset_name in ['Minesweeper', 'Tolokers', 'Questions']:  # Heterophilous datasets
        metric = 'roc'
        dataset = HeterophilousGraphDataset(**dataset_params)
    else:
        metric = 'accuracy'
        dataset = globals()[dataset_name](**dataset_params)

    # Run the model once
    #result = xgb_model(
    #    dataset,
    #    radius1=radius1,
    #    radius2=radius2,
    #    ECT_TYPE='points',
    #    NUM_THETAS=32,
    #    DEVICE='cpu',
    #    metric=metric,
    #    subsample_size=512,
    #)

    result = xgb_model_minibatch(
        dataset,
        radius1=radius1,
        radius2=radius2,
        ECT_TYPE='points',
        NUM_THETAS=32,
        DEVICE='cpu',
        metric=metric,
        num_epochs=5,
        batch_size=64,
        subsample_size=64,
    )

    return result


def run_xgb_model_n_times(dataset_name, dataset_params, radius1, radius2, num_runs):
    # Run the model multiple times in parallel and collect the results
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_xgb_model_once, dataset_name, dataset_params, radius1, radius2)
            for _ in range(num_runs)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Calculate mean and standard deviation of the results
    avg_result = mean(results)
    max_result = max(results)
    std_result = stdev(results)

    return max_result, avg_result, std_result


def main():
    results = {}

    # Use ThreadPoolExecutor to parallelize the dataset configurations
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare the tasks for each configuration to be run 5 times
        future_to_dataset = {
            executor.submit(run_xgb_model_n_times, name, params, r1, r2, num_runs): (name, r1, r2)
            for name, params in datasets
            for r1, r2 in radius_combinations
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_dataset):
            key = future_to_dataset[future]
            key = "-".join(str(x) for x in key)
            try:
                max_result, avg_result, std_result = future.result()
                # Store results as a dictionary with average and standard deviation
                results[key] = {'max': max_result, 'average': avg_result, 'std_dev': std_result}
                print(f"Completed: {key} -> avg: {avg_result}, std: {std_result}")
            except Exception as e:
                print(f"Error with {key}: {e}")

    print(json.dumps(results, indent=4))

    # Dump results to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to 'results.json'")


if __name__ == '__main__':
    main()
