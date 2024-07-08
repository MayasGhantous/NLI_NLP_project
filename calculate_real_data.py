import time
import os
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import tqdm
import language_tool_python



CALCULATE_THE_ERROR_FEATURE = True
REAL_ERROR_FEATURE_LOCATION = 'calculated_data\\real_spell_error_feature.npy'
UNIGRAM_LOCATOIN = 'calculated_data\\Unigrams'
ERROR_HELPER_LOCATION = "calculated_data\\error_count"
FUNCTION_WORDS_ARRAY_LOCATION = 'calculated_data\\function_words.npy'

FUNCTION_WORDS_ARRAY_LOCATION = 'calculated_data\\function_words_features'

SENTENCE_LENGTH_LOCATION = 'average_sentence_lengths.npy'

TRIGRAM_LOCATION = 'calculated_data\\charracter_trigrams'  
def read_files_from_directory(directory):
    try:
        files = defaultdict(list,[])
        i=0
        for country in os.listdir(directory):
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

def get_top_errors(top=400):
    errors = Counter()
    for error_file in os.listdir(ERROR_HELPER_LOCATION):
        if error_file.startswith('all_errors'):
            current = np.load(os.path.join(ERROR_HELPER_LOCATION, error_file), allow_pickle=True).item()
            errors += current
    return errors.most_common(top)
                
            


def get_the_error_feature():
    top_errorss = get_top_errors()
    error_keys = np.array(top_errorss)[:,0]
    new_values = defaultdict(list,[])
    for error_file in os.listdir(ERROR_HELPER_LOCATION):
        if error_file.startswith('indvisual_errors'):
            print(f"Calculating the error feature for {error_file}")
            current = np.load(os.path.join(ERROR_HELPER_LOCATION, error_file), allow_pickle=True).item()
            for usrer_key in current.keys():  
                current_new_values = defaultdict(list,[])
                for listx in current[usrer_key]:
                    current_new_values[usrer_key].append( [listx.get(error_key, 0) for error_key in error_keys])
                new_values.update(current_new_values)
    return new_values

def get_the_function_words_feature():
    function_words_feature = defaultdict(list,[])
    print("getting the function words feature")
    for function_word_file in os.listdir(FUNCTION_WORDS_ARRAY_LOCATION):
        print(f"getting feature for {function_word_file}")
        current = np.load(os.path.join(FUNCTION_WORDS_ARRAY_LOCATION, function_word_file), allow_pickle=True).item()
        function_words_feature.update(current)
    return function_words_feature

def get_the_Unigram_feature():
    unigram_feature = defaultdict(list,[])
    print("getting the unigram feature")
    for unigram_file in os.listdir(UNIGRAM_LOCATOIN):
        print(f"getting feature for {unigram_file}")
        current = np.load(os.path.join(UNIGRAM_LOCATOIN, unigram_file), allow_pickle=True).item()
        unigram_feature.update(current)
    return unigram_feature

def get_the_sentence_length_feature():
    sentence_length_feature = defaultdict(list,[])
    print("getting the sentence length feature")
    
    current = np.load(SENTENCE_LENGTH_LOCATION, allow_pickle=True).item()
    for key in current.keys():
        for value in current[key]:
            sentence_length_feature[key].append([value])
        
    return sentence_length_feature

def get_trigram_Feature():
    trigram_feature = defaultdict(list,[])
    print("getting the trigram feature")
    
    for trigram_file in os.listdir(TRIGRAM_LOCATION):
        print(f"getting feature for {trigram_file}")
        current = np.load(os.path.join(TRIGRAM_LOCATION, trigram_file), allow_pickle=True).item()
        trigram_feature.update(current)
    return trigram_feature


'''
time1 = time.time()
print("calculate the error feature")
new_values = calculate_the_error_feature()
print("Saving the error feature")
np.save(REAL_ERROR_FEATURE_LOCATION, np.array(new_values))
time2 = time.time() 
print(f"Time taken to calculate the error feature: {time2-time1}")
'''
#calculate_the_function_words_feature()
#get_trigram_Feature()