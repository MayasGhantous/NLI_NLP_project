import numpy as np 
from sklearn.model_selection import train_test_split

import Levenshtein as lev

def get_edit_operations(word1, word2):
    # Get the edit operations needed to transform word1 into word2
    operations = lev.editops(word1, word2)
    
    # Print the operations
    for op in operations:
        print(f"Operation: {op[0]}, Source Index: {op[1]}, Target Index: {op[2]}")
        
    return operations

# Example usage
word1 = "mispelled"
word2 = "misspelled"
edit_operations = get_edit_operations(word1, word2)

print(edit_operations)
EDIT_DISTANCE_LOCATION = 'calculated_data\\edit_distance\\reddit.France.txt.tok.clean.npy'
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
Y = [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")

araay = np.load(EDIT_DISTANCE_LOCATION, allow_pickle=True).item()
print(araay)

