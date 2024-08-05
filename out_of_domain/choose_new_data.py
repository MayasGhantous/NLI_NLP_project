import os
import random
import shutil

def remove_folder_and_contents(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Successfully removed the directory and its contents: {directory_path}")
    except Exception as e:
        print(f"Error: {e}")

# Specify the directory path
directory_path = 'C:\\Users\\update\\Documents\\GitHub\\NLP_project\\out_of_domain\\non_europe_data'



def pick_random_folders(directory, num_folders=20):
    # Get a list of all items in the directory
    all_items = os.listdir(directory)
    
    # Filter the list to include only directories
    all_folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    
    # If there are fewer than the desired number of folders, return all of them
    if len(all_folders) <= num_folders:
        return 
    
    # Randomly pick the specified number of folders
    random_folders = random.sample(all_folders, len(all_folders)-num_folders)
    for folder in random_folders:
        folder_path = os.path.join(directory, folder)
        print(f"removeing folder: {folder_path}")    
        remove_folder_and_contents(folder_path)

if __name__ == "__main__":
    for Countery in os.listdir(directory_path):
        pick_random_folders(os.path.join(directory_path, Countery), num_folders=5)
