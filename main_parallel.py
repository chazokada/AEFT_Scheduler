import os
import pandas as pd
import pickle
import glob
from multiprocessing import Pool, cpu_count

from heft import HEFT
from randomHEFT import randomHEFT
from ipeft import IPEFT
from aeft import AEFT
from read_dag import read_dag

# Code written with help from ChatGPT

NUM_TRIALS = 3

# Define the parameter grids for ccr, b, and p
ccr_values = [0.1, 0.5, 1, 5]
b_values = [0.1, 0.5, 1]
p_values = [4, 8, 16]

def evaluate_module_on_file(module, inputs, ccr, b, p):
    """Evaluate the given module on the specified file and parameters."""
    try:
        result = module(input_list=inputs).makespan  # Replace with actual module logic
        return (ccr, b, p, result)  # Return ccr, b, p, and the result
    except Exception as e:
        return (ccr, b, p, f"Error: {str(e)}")

def process_file(file_path, modules):
    """Process a file with multiple modules and return the results for all parameter configurations."""
    results = []

    # Extract the parameters from the filename
    filename_parts = file_path.split('/')[1].split('.dot')[0].split('_')
    n, fat, density, regularity, jump = filename_parts

    # Loop over all parameter combinations for ccr, b, and p
    param_combinations = [(ccr, b, p) for ccr in ccr_values for b in b_values for p in p_values]
    for ccr, b, p in param_combinations:
        
        # Initialize a dictionary to store the makespans for each trial
        makespans = {module.__name__: [] for module in modules}

        # Run NUM_TRIALS trials
        for _ in range(NUM_TRIALS):
            # Read the DAG once for the current parameter configuration (ensure it's the same for each trial)
            inputs = read_dag(file_path, p=p, b=b, ccr=ccr)
            for module in modules:
                result = evaluate_module_on_file(module, inputs, ccr, b, p)
                makespans[module.__name__].append(result[3])

        # Calculate the average makespan for each module over NUM_TRIALS
        for module in modules:
            average_makespan = sum(makespans[module.__name__]) / NUM_TRIALS
            results.append({
                "file_path": file_path,
                "n": n,
                "fat": fat,
                "density": density,
                "regularity": regularity,
                "jump": jump,
                "ccr": ccr,
                "b": b,
                "p": p,
                "module": module.__name__,
                "average_makespan": average_makespan  # Average makespan over NUM_TRIALS
            })

    return results

def process_files_in_parallel(data_files, modules):
    """Process all files in parallel using multiple CPUs."""
    all_results = []

    with Pool(cpu_count()) as pool:
        # Use starmap to send both file_path and modules to process_file
        results = pool.starmap(process_file, [(file_path, modules) for file_path in data_files])

        # Flatten the list of results
        all_results = [result for sublist in results for result in sublist]

    return all_results

def save_results_to_pickle(results, output_pickle_path):
    """Save all results to the pickle file."""
    df = pd.DataFrame(results)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(df, f)

def main(data_folder, modules, output_pickle_path):
    """Main function to process all files and save results."""
    # Find all .dot files in the specified folder
    data_files = glob.glob(os.path.join(data_folder, "*.dot"))

    if not data_files:
        print("No .dot files found in the specified folder.")
        return

    print(f"Starting parallel processing with {cpu_count()} CPUs.")
    
    # Process all files in parallel and get results
    all_results = process_files_in_parallel(data_files, modules)

    # Save accumulated results to pickle after all files are processed
    save_results_to_pickle(all_results, output_pickle_path)

    print(f"All results processed and saved to {output_pickle_path}")

if __name__ == "__main__":
    # Folder containing .dot files
    data_folder = "dag"  # Folder with your .dot files

    # List of modules to evaluate on the files
    modules = [HEFT, IPEFT, AEFT]  # Replace with actual modules

    # Output pickle file path
    output_pickle_path = "evaluation_results.pkl"

    # Run the main function
    main(data_folder, modules, output_pickle_path)
