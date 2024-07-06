import numpy as np 
WHAT_COUNTRY = 0
ERRORS_LOCATION = f'C:\\Users\\update\\Documents\\GitHub\\NLP_project\\calculated_data\\grammer_errors\\reddit.Mexico.txt.tok.clean.npy'
ALL_ERRORS_LOCATION = f'C:\\Users\\update\\Documents\\GitHub\\NLP_project\\calculated_data\\error_count\\all_errors{WHAT_COUNTRY}.npy'
array = np.load(ERRORS_LOCATION,allow_pickle=True).item()
print(array)
