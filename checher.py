import numpy as np 
from sklearn.model_selection import train_test_split
'''
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
Y = [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")'''
import language_tool_python
import torch
print(torch.cuda.get_device_name(torch.cuda.current_device()))
# Initialize the tool for the English language
'''
tool = language_tool_python.LanguageTool('en-US')

# Text to be checked
text = "This is an exampel of text with a speling mistake."

# Check the text for errors
matches = tool.check(text)

# Display the errors found
for match in matches:
    print(f"Error: {match.message}")
    print(f"Suggestion: {match.replacements}")
    print(f"Context: {text[match.offset:match.offset+match.errorLength]}")
    print(f"Rule ID: {match.ruleId}\n")

# Optionally, you can apply the corrections automatically
corrected_text = language_tool_python.utils.correct(text, matches)
print("Corrected Text:", corrected_text)

'''