import os
import glob

from aeft import AEFT
from read_dag import read_dag

# Define the parameter grids for ccr, b, and p
ccr_values = [0.1, 0.5, 1, 5]
b_values = [0.1, 0.5, 1]
p_values = [4, 8, 16]
param_combinations = [(ccr, b, p) for ccr in ccr_values for b in b_values for p in p_values]

def main(data_dir, im_dir):
    """Main function to process all files and save images of schedule results."""
    
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    
    # Find all .dot files in the specified folder
    data_files = glob.glob(os.path.join(data_dir, "*.dot"))

    if not data_files:
        print("No .dot files found in the specified folder.")
        return

    for file_name in data_files:
        file_root = file_name.split('/')[1].split('.dot')[0]
        # Loop over all parameter combinations for ccr, b, and p
        for ccr, b, p in param_combinations:
            root = file_root + "_" + str(ccr) + "_" + str(b) + "_" + str(p)
            img_title = "Task Schedule: " + root
            save_path = os.path.join(im_dir,root + ".png")
            inputs = read_dag(file_name, p=p, b=b, ccr=ccr)
            aeft = AEFT(input_list=inputs)
            aeft.show_schedule(plt_title=img_title,save_path=save_path)

if __name__ == "__main__":
    # Folder containing .dot files
    data_dir = "dag" 
    # Output image dir
    im_dir = "images"

    # Run the main function
    main(data_dir, im_dir)