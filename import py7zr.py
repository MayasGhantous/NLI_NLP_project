
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
import Levenshtein as lev
# Download the necessary NLTK resources

DATA_LOCATION = 'europe_data'
WHAT_COUNTRY = 30
DO_WE_NEED_TO_EXTRACT_UNIGRAMS = True
UNIGRAM_LOCATOIN = 'calculated_data\\Unigrams2'
DO_WE_NEED_TO_EXTRACT_TOP_UNIGRAMS = True
Top_UNIGRAM_LOCATOIN = 'top_unigrams2.npy'
DO_WE_NEED_TO_EXTRACT_TOP_ERRORS = False
ALL_ERRORS_LOCATION = f'calculated_data\\error_count2\\all_errors{WHAT_COUNTRY}.npy'
ERRORS_LOCATION = f'calculated_data\\error_count2\\indvisual_errors{WHAT_COUNTRY}.npy'

DO_WE_NEED_TO_SET_FUNCTION_WORDS = False
FUNCTION_WORDS_ARRAY_LOCATION = 'calculated_data\\function_words.npy'
FUNCTION_WORDS_TEXT_LOCATION = 'function_words.txt'
DO_WE_NEED_TOCALCULATE_FUNCTION_WORDS_FEATURE = False
FUNCTOIN_WORDS_FEATURE_LOCATION = 'calculated_data\\function_words_features'
DO_WE_NEED_TO_EXTARCT_EDIT_DISTANCE = False
EDIT_DISTANCE_LOCATION = 'calculated_data\\edit_distance'



GET_ALL_FIELS = False
def read_files_from_directory(directory):
    global WHAT_COUNTRY
    global DO_WE_NEED_TO_EXTRACT_UNIGRAMS
    global DO_WE_NEED_TO_EXTRACT_TOP_ERRORS
    global DO_WE_NEED_TO_SET_FUNCTION_WORDS
    global DO_WE_NEED_TOCALCULATE_FUNCTION_WORDS_FEATURE
    global DATA_LOCATION
    global UNIGRAM_LOCATOIN
    global ALL_ERRORS_LOCATION
    global FUNCTION_WORDS_ARRAY_LOCATION
    global FUNCTOIN_WORDS_FEATURE_LOCATION
    global ERRORS_LOCATION
    global FUNCTION_WORDS_TEXT_LOCATION
    global DO_WE_NEED_TO_EXTRACT_TOP_UNIGRAMS
    global Top_UNIGRAM_LOCATOIN
    global GET_ALL_FIELS
    global DO_WE_NEED_TO_EXTARCT_EDIT_DISTANCE
    global EDIT_DISTANCE_LOCATION
    try:
        files = defaultdict(list,[])
        i=0
        for country in os.listdir(directory):
            if i != WHAT_COUNTRY and GET_ALL_FIELS == False:
                i+=1
                continue
            i+=1
            if DO_WE_NEED_TOCALCULATE_FUNCTION_WORDS_FEATURE:
                FUNCTOIN_WORDS_FEATURE_LOCATION += "\\"+country+".npy"
            if DO_WE_NEED_TO_EXTRACT_UNIGRAMS:
                UNIGRAM_LOCATOIN += "\\"+country+'.npy'
            if DO_WE_NEED_TO_EXTRACT_TOP_ERRORS:
                ALL_ERRORS_LOCATION += country+'.npy'
                ERRORS_LOCATION += country+'.npy'
            if DO_WE_NEED_TO_EXTARCT_EDIT_DISTANCE:
                EDIT_DISTANCE_LOCATION += "\\"+country+'.npy'


            #ALL_ERRORS_LOCATION 
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
    #stop_words = set(stopwords.words('english'))
    #words = [word for word in words if word.isalnum() and word not in stop_words]
    return words


def extract_top_unigrams(text, top_n=1000):
    # Tokenize the text into words
    word_counts = Counter()
    print('geting top unigrams')
    for key in tqdm.tqdm(text.keys()):
        for content in text[key]:
            words = get_chunk_tokens(content)
            
            
            # Count word frequencies
            word_counts+=Counter(words)
            
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
            unigram_feature[key].append([ chunk_tokens_counter[unigram] / total_tokens for unigram, _ in top_unigrams])
    return unigram_feature

#speller = enchant.Dict("en_US")
tool = language_tool_python.LanguageTool('en-US')

def get_all_edits(content,chunk_error):
    edits = Counter()
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
        '''i = 0
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
        '''
        current_edit = lev.editops(misspelled_word, corrected_word)
        for edit in current_edit:
            if edit[0] == 'replace':
                edits[f'sub({misspelled_word[edit[1]]},{corrected_word[edit[2]]})'] += 1
            elif edit[0] == 'insert':
                edits[f'ins({corrected_word[edit[2]]})'] += 1
            elif edit[0] == 'delete':
                edits[f'del({misspelled_word[edit[1]]})'] += 1

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
            errors_dic[key].append(edits)
            overall_errors+=edits
        
    
    saveing_array = np.array(errors_dic)
    np.save(ERRORS_LOCATION, saveing_array)
    most_common  = overall_errors
    return most_common

def create_function_words_array(): 
    with open(FUNCTION_WORDS_TEXT_LOCATION, 'r') as file:
        function_words = file.read().split()
    np.save(FUNCTION_WORDS_ARRAY_LOCATION, function_words)

def calculate_function_words_feature(dic, function_words):
    function_words_feature = defaultdict(list,[])
    print("Calculating the function words feature")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            words = word_tokenize(content)
            chunk_tokens = [word.lower() for word in words]
            #chunk_tokens = get_chunk_tokens(content)
            chunk_tokens_counter = Counter(chunk_tokens)
            function_words_feature[key].append([chunk_tokens_counter[function_word] for function_word in function_words])
    return function_words_feature


def calulate_ditsance(content, chunk_errors):
    distence = 0
    for error in chunk_errors:
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
        distence += lev.distance(misspelled_word, corrected_word) 
    return distence/len(word_tokenize(content))





def calculate_edit_distance(dic):
    edit_distance_feature = defaultdict(list,[])
    print("Calculating the edit distance feature")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            chunk_erroes = find_errors(content)
            distance = calulate_ditsance(content,chunk_erroes)
            edit_distance_feature[key].append([distance])
        
    return edit_distance_feature
    


def main():
    global WHAT_COUNTRY
    global DO_WE_NEED_TO_EXTRACT_UNIGRAMS
    global DO_WE_NEED_TO_EXTRACT_TOP_ERRORS
    global DO_WE_NEED_TO_SET_FUNCTION_WORDS
    global DO_WE_NEED_TOCALCULATE_FUNCTION_WORDS_FEATURE
    global DATA_LOCATION
    global UNIGRAM_LOCATOIN
    global ALL_ERRORS_LOCATION
    global FUNCTION_WORDS_ARRAY_LOCATION
    global FUNCTOIN_WORDS_FEATURE_LOCATION
    global ERRORS_LOCATION
    global FUNCTION_WORDS_TEXT_LOCATION
    global DO_WE_NEED_TO_EXTRACT_TOP_UNIGRAMS
    global Top_UNIGRAM_LOCATOIN
    global GET_ALL_FIELS
    global DO_WE_NEED_TO_EXTARCT_EDIT_DISTANCE
    global EDIT_DISTANCE_LOCATION
    nltk.download('punkt')
    nltk.download('stopwords') 
    time_start = time.time()
    
    if DO_WE_NEED_TO_EXTRACT_TOP_UNIGRAMS:
        GET_ALL_FIELS = True
        files = None
        files = read_files_from_directory(DATA_LOCATION)
        top_unigrams = extract_top_unigrams(files)
        top_unigrams = np.array(top_unigrams)
        np.save(Top_UNIGRAM_LOCATOIN, top_unigrams)
        GET_ALL_FIELS = False
    else:
        top_unigrams = np.load(Top_UNIGRAM_LOCATOIN, allow_pickle=True)

    if DO_WE_NEED_TO_EXTRACT_UNIGRAMS:
        list_of_numbers =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
        for i in list_of_numbers:
            WHAT_COUNTRY = i
            UNIGRAM_LOCATOIN = 'calculated_data\\Unigrams2'
            files = read_files_from_directory(DATA_LOCATION)
            unigram_feature = calculate_the_unigram_feature(files, top_unigrams)
            unigram_feature = np.array(unigram_feature)
            np.save(UNIGRAM_LOCATOIN, unigram_feature)



    if DO_WE_NEED_TO_EXTRACT_TOP_ERRORS:
        list_of_numbers =[6,7,8,9,10,11,12,13,14]
        for i in list_of_numbers:
            WHAT_COUNTRY = i
            ALL_ERRORS_LOCATION = f'calculated_data\\error_count2\\all_errors_'
            ERRORS_LOCATION = f'calculated_data\\error_count2\\indvisual_errors_'
            files = read_files_from_directory(DATA_LOCATION)
            top_errors = get_top_errors(files)
            top_errors = np.array(top_errors)
            np.save(ALL_ERRORS_LOCATION, top_errors)
            print(top_errors)

    if DO_WE_NEED_TO_SET_FUNCTION_WORDS:
        create_function_words_array()

    if DO_WE_NEED_TOCALCULATE_FUNCTION_WORDS_FEATURE:
        function_words = np.load(FUNCTION_WORDS_ARRAY_LOCATION, allow_pickle=True)
        list_of_numbers =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
        for i in list_of_numbers:
            WHAT_COUNTRY = i
            FUNCTOIN_WORDS_FEATURE_LOCATION = 'calculated_data\\function_words_features'
            files = read_files_from_directory(DATA_LOCATION)
            function_words_feature = calculate_function_words_feature(files, function_words)
            function_words_feature = np.array(function_words_feature)
            np.save(FUNCTOIN_WORDS_FEATURE_LOCATION, function_words_feature)

    if DO_WE_NEED_TO_EXTARCT_EDIT_DISTANCE:
        list_of_numbers =[8,9,10,11,12,13,14]
        for i in list_of_numbers:
            WHAT_COUNTRY = i
            EDIT_DISTANCE_LOCATION = 'calculated_data\\edit_distance'
            files = read_files_from_directory(DATA_LOCATION)
            edit_distance_feature = calculate_edit_distance(files)
            edit_distance_feature = np.array(edit_distance_feature)
            np.save(EDIT_DISTANCE_LOCATION, edit_distance_feature)

        

    #unigram_feature = calculate_the_unigram_feature(files, top_unigrams)
    time_end = time.time()
    print(f"Top unigrams: {top_unigrams}")
    print(f"Time taken to read data: {time_end - time_start} seconds")
    time_start = time.time()
if __name__ == "__main__":
    main() 