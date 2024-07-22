from sklearn.svm import SVC 
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from calculate_real_data_out_of_domain import get_the_error_feature,get_the_function_words_feature,get_the_Unigram_feature
from calculate_real_data_out_of_domain import get_the_sentence_length_feature, get_trigram_Feature,get_the_grammer_feature
from calculate_real_data_out_of_domain import get_pos_trigram,get_CharTrigram_Tokens_Unigram_Spelling_Feature, get_Function_words_Pos_Trigram_Sentence_length_Feature
from calculate_real_data_out_of_domain import get_grammer_spelling_features


import os
s = os.path.abspath('..')
sys.path.append(os.path.join(s,'NLP_project'))
'''
from calculate_real_data import get_the_error_feature,get_the_function_words_feature,get_the_Unigram_feature
from calculate_real_data import get_the_sentence_length_feature, get_trigram_Feature,get_the_grammer_feature
from calculate_real_data import get_pos_trigram,get_CharTrigram_Tokens_Unigram_Spelling_Feature, get_Function_words_Pos_Trigram_Sentence_length_Feature
from calculate_real_data import get_grammer_spelling_features
'''
from classification import create_NLI,create_binary,cereate_family,MODELS_LOCATION
import pandas as pd
from scipy.sparse import csr_matrix
from imblearn.under_sampling import RandomUnderSampler
import pickle



DO_WE_NEED_TO_TEST_NLI = True
DO_WE_NEED_TO_TEST_BINARY = True
DO_WE_NEED_TO_TEST_FAMILY = True

if __name__ == '__main__':
    FEATURE_NAME = 'CharTrigram_Tokens_Unigram_Spelling'
    feature = get_CharTrigram_Tokens_Unigram_Spelling_Feature()
    print(f"Feature {FEATURE_NAME} loaded")
    if DO_WE_NEED_TO_TEST_NLI:
        print('Testing NLI')
        SVM_location = FEATURE_NAME + '_NLI_model.pkl'
        SVM_location = os.path.join(MODELS_LOCATION, SVM_location)

        with open(SVM_location, 'rb') as file:
            model = pickle.load(file)
            print("model loaded")
            X, y = create_NLI(feature)
            rus = RandomUnderSampler(random_state=42)
            #X, y= rus.fit_resample(X, y)
            X = pd.DataFrame(X)
            X_test = csr_matrix(X)
            print(X.shape)
            y_pred = model.predict(X_test)
            print("NLI model")
            print(accuracy_score(y, y_pred))
if DO_WE_NEED_TO_TEST_BINARY:
    print('Testing Binary')
    SVM_location = FEATURE_NAME + '_binary_model.pkl'
    SVM_location = os.path.join(MODELS_LOCATION, SVM_location)
    with open(SVM_location, 'rb') as file:
        model = pickle.load(file)
        print("model loaded")
        X, y = create_binary(feature)
        rus = RandomUnderSampler(random_state=42)
        #X, y= rus.fit_resample(X, y)
        X = pd.DataFrame(X)
        X_test = csr_matrix(X)
        print(X.shape)
        y_pred = model.predict(X_test)
        print("Binary model")
        print(accuracy_score(y, y_pred))
if DO_WE_NEED_TO_TEST_FAMILY:
    print('Testing Family')
    SVM_location = FEATURE_NAME + '_family_model.pkl'
    SVM_location = os.path.join(MODELS_LOCATION, SVM_location)
    with open(SVM_location, 'rb') as file:
        model = pickle.load(file)
        print("model loaded")
        X, y = cereate_family(feature)
        rus = RandomUnderSampler(random_state=42)
        #X, y= rus.fit_resample(X, y)
        X = pd.DataFrame(X)
        X_test = csr_matrix(X)
        print(X.shape)
        y_pred = model.predict(X_test)
        print("Family model")
        print(accuracy_score(y, y_pred))
        
