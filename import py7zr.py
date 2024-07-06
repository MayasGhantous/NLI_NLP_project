
#helper to calculate the needed data for the project(top unigrams and top errors)
import time
import os
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import tqdm
import pickle
import language_tool_python
from multiprocessing import Pool, cpu_count
# Download the necessary NLTK resources

WHAT_COUNTRY = 29
DO_WE_NEED_TO_EXTRACT_UNIGRAMS = False
UNIGRAM_LOCATOIN = 'top_unigrams.npy'
DO_WE_NEED_TO_EXTRACT_TOP_ERRORS = True
ALL_ERRORS_LOCATION = f'C:\\Users\\update\\Documents\\GitHub\\NLP_project\\calculated_data\\error_count\\all_errors{WHAT_COUNTRY}.npy'
ERRORS_LOCATION = f'C:\\Users\\update\\Documents\\GitHub\\NLP_project\\calculated_data\\error_count\\indvisual_errors{WHAT_COUNTRY}.npy'
def read_files_from_directory(directory):
    try:
        files = defaultdict(list,[])
        i=0
        for country in os.listdir(directory):
            if i != WHAT_COUNTRY:
                i+=1
                continue
            i+=1
            
            print(f"{i}: Reading files from {country}")
            country_path = os.path.join(directory, country)
            if os.path.isfile(country) == False:
                for person in os.listdir(country_path):
                    person_path = os.path.join(country_path, person)
                    for person_chunk in os.listdir(person_path):
                        file_path = os.path.join(person_path, person_chunk)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            files[(country,person)].append(text)
        return files

                
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_chunk_tokens(text):
    words = word_tokenize(text)
    
    # Convert to lowercase
    words = [word.lower() for word in words]
    
    # Remove punctuation and stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words


def extract_top_unigrams(text, top_n=1000):
    # Tokenize the text into words
    word_counts = Counter()
    for key in text.keys():
        for content in text[key]:
            words = get_chunk_tokens(content)
            
            
            # Count word frequencies
            word_counts.update(words)
            
    # Get the top N unigrams
    top_unigrams = word_counts.most_common(top_n)
    
    return top_unigrams


def calculate_the_unigram_feature(dic , top_unigrams):
    unigram_feature = defaultdict(list,[])
    print("Calculating the unigram feature")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            chunk_tokens = get_chunk_tokens(content)
            chunk_tokens_counter = Counter(chunk_tokens)
            total_tokens = len(chunk_tokens)
            unigram_feature[key].append({unigram: chunk_tokens_counter[unigram] / total_tokens for unigram, _ in top_unigrams})
    return unigram_feature

#speller = enchant.Dict("en_US")
tool = language_tool_python.LanguageTool('en-US')

def get_all_edits(content,chunk_error):
    edits = []
    for error in chunk_error:
        error_start = error.offset
        error_end = error.offset + error.errorLength
        misspelled_word = content[error_start:error_end]
        if error.replacements == None:
            continue
        if len(error.replacements) == 0:
            continue
        corrected_word = error.replacements[0]
        misspelled_word = misspelled_word.lower()
        corrected_word = corrected_word.lower()
        if misspelled_word == corrected_word:
            continue
        #print(f"Misspelled word: {misspelled_word}, Corrected word: {corrected_word}")
        if corrected_word == None:
            continue
        i = 0
        j = 0
        while i < len(misspelled_word) and j < len(corrected_word):
            if misspelled_word[i] != corrected_word[j]:
                if len(misspelled_word) > len(corrected_word):
                    edits.append(f'del({misspelled_word[i]})')
                    i += 1
                elif len(misspelled_word) < len(corrected_word):
                    edits.append(f'ins({corrected_word[j]})')
                    j += 1
                else:
                    edits.append(f'sub({misspelled_word[i]},{corrected_word[j]})')
                    i += 1
                    j += 1
            else:
                i += 1
                j += 1
        while i < len(misspelled_word):
            edits.append(f'del({misspelled_word[i]})')
            i += 1
        while j < len(corrected_word):
            edits.append(f'ins({corrected_word[j]})')
            j += 1
    return edits

def find_errors(content):
    matches = tool.check(content)
    errors = [match for match in matches if match.ruleId == "MORFOLOGIK_RULE_EN_US"]
    return errors

def prcoess (content):
    chunk_errors = find_errors(content)
    edits = get_all_edits(content,chunk_errors)
    return edits

def get_top_errors(dic, top_errors = 400):
    print("Calculating the top errors")
    overall_errors = Counter()
    errors_dic = defaultdict(list,[])
    l=0
    for key in tqdm.tqdm(dic.keys()):

    
        for content in dic[key]:
            chunk_errors = find_errors(content)
            edits = get_all_edits(content,chunk_errors)
            errors_dic[key].append(Counter(edits))
            overall_errors+=Counter(edits)
        
    
    saveing_array = np.array(errors_dic)
    np.save(ERRORS_LOCATION, saveing_array)
    most_common  = overall_errors
    return most_common

def calculate_spelling_errors(dic,top_errors):
    pass
    


def main():
    nltk.download('punkt')
    nltk.download('stopwords') 
    time_start = time.time()
    files = None
    files = read_files_from_directory('C:\\Users\\update\\Documents\\GitHub\\NLP_project\\europe_data')
    if DO_WE_NEED_TO_EXTRACT_UNIGRAMS:
        top_unigrams = extract_top_unigrams(files)
        top_unigrams = np.array(top_unigrams)
        np.save(UNIGRAM_LOCATOIN, top_unigrams)
    else:
        top_unigrams = np.load(UNIGRAM_LOCATOIN, allow_pickle=True)

    if DO_WE_NEED_TO_EXTRACT_TOP_ERRORS:
        top_errors = get_top_errors(files)
        top_errors = np.array(top_errors)
        np.save(ALL_ERRORS_LOCATION, top_errors)
        print(top_errors)
    else:
        top_errors = np.load(ALL_ERRORS_LOCATION, allow_pickle=True)

    #unigram_feature = calculate_the_unigram_feature(files, top_unigrams)
    time_end = time.time()
    print(f"Top unigrams: {top_unigrams}")
    print(f"Time taken to read data: {time_end - time_start} seconds")
    time_start = time.time()
if __name__ == "__main__":
    main() 