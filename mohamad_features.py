import time
import os
from collections import defaultdict
import nltk
from nltk import pos_tag, ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import tqdm
import language_tool_python

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
WHAT_COUNTRY = 1
DO_WE_NEED_TO_EXTRACT_TOP_TRIGRAMS = False
TOP_TRIGRAMS_LOCATION = 'top_trigrams.npy'
DO_WE_NEED_TO_EXTRACT_TRIGRAMS = False
TRIGRAM_LOCATION = 'calculated_data\\charracter_trigrams'
DO_WE_NEED_TO_EXTRACT_GRAMMAR_ERRORS = False
GRAMMAR_ERRORS_LOCATION = 'calculated_data\\grammer_errors'
DO_WE_NEED_TO_EXTRACT_POS_TRIGRAMS = True
POS_TRIGRAMS_LOCATION = 'top_pos_trigrams.npy'
DO_WE_NEED_TO_EXTRACT_SENTENCE_LENGTH = False
SENTENCE_LENGTH_LOCATION = 'average_sentence_lengths.npy'




def read_files_from_directory(directory):
    try:
        global GRAMMAR_ERRORS_LOCATION
        global TRIGRAM_LOCATION
        global WHAT_COUNTRY
        global DO_WE_NEED_TO_EXTRACT_GRAMMAR_ERRORS
        global DO_WE_NEED_TO_EXTRACT_TRIGRAMS
        files = defaultdict(list, [])
        i = 0
        for country in os.listdir(directory):
            '''if i != WHAT_COUNTRY:
                i += 1
                continue'''
            if DO_WE_NEED_TO_EXTRACT_GRAMMAR_ERRORS:
                GRAMMAR_ERRORS_LOCATION += "\\"+country+".npy"
            if DO_WE_NEED_TO_EXTRACT_TRIGRAMS:
                TRIGRAM_LOCATION += "\\"+country+".npy"
            i += 1
            print(f"{i}: Reading files from {country}")
            country_path = os.path.join(directory, country)
            if os.path.isfile(country) == False:
                for person in os.listdir(country_path):
                    person_path = os.path.join(country_path, person)
                    for person_chunk in os.listdir(person_path):
                        file_path = os.path.join(person_path, person_chunk)
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                            files[(country, person)].append(text)
        return files


    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_character_ngrams(text, n=3):
    text = text.replace(" ", "")
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ngrams


def extract_top_trigrams(text, top_n=1000):
    trigram_counts = Counter()
    for key in text.keys():
        for content in text[key]:
            trigrams = get_character_ngrams(content)
            trigram_counts.update(trigrams)
    top_trigrams = trigram_counts.most_common(top_n)
    return top_trigrams


def calculate_trigram_features(dic, top_trigrams):
    trigram_features = defaultdict(list, [])
    print("Calculating the trigram features")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            trigrams = get_character_ngrams(content)
            trigram_counter = Counter(trigrams)
            total_trigrams = len(trigrams)
            trigram_features[key].append([trigram_counter[trigram] / total_trigrams for trigram, _ in top_trigrams])
    return trigram_features


def calculate_average_sentence_length(dic):
    average_sentence_lengths = defaultdict(list, [])
    print("Calculating the average sentence length")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            sentences = nltk.sent_tokenize(content)
            if len(sentences) > 0:
                average_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)
            else:
                average_length = 0
            average_sentence_lengths[key].append(average_length)
    return average_sentence_lengths


def extract_pos_trigrams(text):
    tokens = word_tokenize(text)
    pos_tags = [tag for _, tag in pos_tag(tokens)]
    pos_trigrams = list(ngrams(pos_tags, 3))
    return pos_trigrams


def extract_pos_trigram_features(dic, top_n=300):
    all_trigrams = []
    pos_trigram_features = defaultdict(list)

    print("Extracting POS trigrams")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            trigrams = extract_pos_trigrams(content)
            all_trigrams.extend(trigrams)

    top_trigrams = [trigram for trigram, _ in Counter(all_trigrams).most_common(top_n)]
    np.save('top_300_pos.npy', top_trigrams)

    print("Calculating POS trigram frequencies")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            trigrams = extract_pos_trigrams(content)
            trigram_counts = Counter(trigrams)
            total_trigrams = len(trigrams)

            features = {}
            for trigram in top_trigrams:
                frequency = trigram_counts[trigram] / total_trigrams if total_trigrams > 0 else 0
                features[' '.join(trigram)] = frequency

            pos_trigram_features[key].append(features)

    return pos_trigram_features


def extract_grammar_errors(dic):
    grammar_errors = defaultdict(list, [])
    print("Extracting grammar errors")
    for key in tqdm.tqdm(dic.keys()):
        for content in dic[key]:
            matches = tool.check(content)
            error_features = defaultdict(int)
            for match in matches:
                rule_id = match.ruleId
                error_features[rule_id] += 1
            # Normalize by content length
            content_length = len(content.split())  # Count words
            for rule_id in error_features:
                error_features[rule_id] /= content_length

            grammar_errors[key].append(error_features)
    return grammar_errors


tool = language_tool_python.LanguageTool('en-US')
time_start = time.time()
files = None
files = read_files_from_directory('europe_data')

if DO_WE_NEED_TO_EXTRACT_TOP_TRIGRAMS:
    top_trigrams = extract_top_trigrams(files)
    top_trigrams = np.array(top_trigrams)
    np.save(TOP_TRIGRAMS_LOCATION, top_trigrams)

if DO_WE_NEED_TO_EXTRACT_TRIGRAMS:
    list_of_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    top_trigrams = np.load(TOP_TRIGRAMS_LOCATION, allow_pickle=True)
    for i in list_of_numbers:
        WHAT_COUNTRY = i
        TRIGRAM_LOCATION = f'calculated_data\\charracter_trigrams'
        files1 = read_files_from_directory('europe_data')
        trigrams = calculate_trigram_features(files1,top_trigrams)
        print(f"saving: {WHAT_COUNTRY}")
        np.save(TRIGRAM_LOCATION, np.array(trigrams))


if DO_WE_NEED_TO_EXTRACT_GRAMMAR_ERRORS:
    list_of_numbers = [0,1,2,3,4,5]
    for i in list_of_numbers:
        WHAT_COUNTRY = i
        GRAMMAR_ERRORS_LOCATION = f'calculated_data\\grammer_errors'
        files1 = read_files_from_directory('europe_data')
        grammar_errors = extract_grammar_errors(files1)
        print(f"saving: {WHAT_COUNTRY}")
        np.save(GRAMMAR_ERRORS_LOCATION, np.array(grammar_errors))


if DO_WE_NEED_TO_EXTRACT_SENTENCE_LENGTH:
    average_sentence_lengths = calculate_average_sentence_length(files)
    average_sentence_lengths = np.array(average_sentence_lengths)
    np.save(SENTENCE_LENGTH_LOCATION, average_sentence_lengths)
else:
    average_sentence_lengths = np.load("average_sentence_lengths.npy", allow_pickle=True)

if DO_WE_NEED_TO_EXTRACT_POS_TRIGRAMS:
    pos_trigram_features = extract_pos_trigram_features(files)
    np.save(POS_TRIGRAMS_LOCATION, pos_trigram_features)
else:
    pos_trigram_features = np.load(POS_TRIGRAMS_LOCATION, allow_pickle=True)

time_end = time.time()
#print(f"Top trigrams: {top_trigrams}")
print(f"Grammar errors: {grammar_errors}")
#print(f"sentence length: {average_sentence_lengths}")
#print(f"pos trigrams: {pos_trigram_features}")
print(f"Time taken to read data: {(time_end - time_start)/60} minutes")
time_start = time.time()