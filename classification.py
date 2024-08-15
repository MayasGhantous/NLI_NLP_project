from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from calculate_real_data import get_the_error_feature,get_the_function_words_feature,get_the_Unigram_feature
from calculate_real_data import get_the_sentence_length_feature, get_trigram_Feature,get_the_grammer_feature
from calculate_real_data import get_pos_trigram,get_CharTrigram_Tokens_Unigram_Spelling_Feature, get_Function_words_Pos_Trigram_Sentence_length_Feature
from calculate_real_data import get_grammer_spelling_features
import pandas as pd
from scipy.sparse import csr_matrix
from imblearn.under_sampling import RandomUnderSampler
import pickle

English = ['Australia','UK','US','NewZealand','Ireland']
German = ['Austria', 'Germany']
Bulgarian = ['Bulgaria']
Croatian = ['Croatia']
Czech = ['Czech']
Estonian = ['Estonia']#
Finnish = ['Finland']#
French = ['France']
Greek = ['Greece']#
Hungarian = ['Hungary']#
Italian = ['Italy']
Lithuanian = ['Lithuania']#
Spanish = ['Mexico', 'Spain']
Dutch = ['Netherlands']
Norwegian = ['Norway']
Polish = ['Poland']
Portuguese = ['Portugal']
Romanian = ['Romania']
Russian = ['Russia']
Serbian = ['Serbia']
Slovenian = ['Slovenia']
Swedish = ['Sweden']
Turkish = ['Turkey']#
MODELS_LOCATION ='svm_moduels'
DO_WE_NEED_NLI_MODEL = True
DO_WE_NEED_BINARY_MODEL = True
DO_WE_NEED_FAMILY_MODEL = True

def create_NLI(feature):
    y = []
    X = []
    for key in feature.keys():
        lablel = key[0][7:-14]
        if lablel == 'Ukraine':
            continue
        if lablel in English:
            y.extend([0 for _ in range(len(feature[key]))])
        elif lablel in German:
            y.extend([1 for _ in range(len(feature[key]))])
        elif lablel in Bulgarian:
            y.extend([2 for _ in range(len(feature[key]))])
        elif lablel in Croatian:
            y.extend([3 for _ in range(len(feature[key]))])
        elif lablel in Czech:
            y.extend([4 for _ in range(len(feature[key]))])
        elif lablel in Estonian:
            y.extend([5 for _ in range(len(feature[key]))])
        elif lablel in Finnish:
            y.extend([6 for _ in range(len(feature[key]))])
        elif lablel in French:
            y.extend([7 for _ in range(len(feature[key]))])
        elif lablel in Greek:
            y.extend([8 for _ in range(len(feature[key]))])
        elif lablel in Hungarian:
            y.extend([9 for _ in range(len(feature[key]))])
        elif lablel in Italian:
            y.extend([10 for _ in range(len(feature[key]))])
        elif lablel in Lithuanian:
            y.extend([11 for _ in range(len(feature[key]))])
        elif lablel in Spanish:
            y.extend([12 for _ in range(len(feature[key]))])
        elif lablel in Dutch:
            y.extend([13 for _ in range(len(feature[key]))])
        elif lablel in Norwegian:
            y.extend([14 for _ in range(len(feature[key]))])
        elif lablel in Polish:
            y.extend([15 for _ in range(len(feature[key]))])
        elif lablel in Portuguese:
            y.extend([16 for _ in range(len(feature[key]))])
        elif lablel in Romanian:
            y.extend([17 for _ in range(len(feature[key]))])
        elif lablel in Russian:
            y.extend([18 for _ in range(len(feature[key]))])
        elif lablel in Serbian:
            y.extend([19 for _ in range(len(feature[key]))])
        elif lablel in Slovenian:
            y.extend([20 for _ in range(len(feature[key]))])
        elif lablel in Swedish:
            y.extend([21 for _ in range(len(feature[key]))])
        elif lablel in Turkish:
            y.extend([22 for _ in range(len(feature[key]))])
        else:
            print(f"Error: {lablel}")
            continue
        X.extend(feature[key])
    return X, y

def create_binary(feature):
    y = []
    X = []
    for key in feature.keys():
        lablel = key[0][7:-14]
        if lablel == 'Ukraine':
            continue
        if lablel in English:
            y.extend([0 for _ in range(len(feature[key]))])
        else:
            y.extend([1 for _ in range(len(feature[key]))])
        X.extend(feature[key])
    return X, y

def cereate_family(feature):
    Native_English = ['Australia','UK','US','NewZealand','Ireland']
    Germanic = ['Austria', 'Germany','Netherlands','Norway','Sweden']
    Slavic = ['Bulgaria','Croatia','Czech','Poland','Russia','Serbia','Slovenia','Lithuania']
    Romance = ['France','Italy','Mexico','Portugal','Spain','Romania']
    others = ['Estonia','Finland','Greece','Hungary','Turkey']
    y = []
    X = []
    for key in feature.keys():
        lablel = key[0][7:-14]
        if lablel == 'Ukraine':
            continue
        if lablel in Native_English:
            y.extend([0 for _ in range(len(feature[key]))])
        elif lablel in Germanic:
            y.extend([1 for _ in range(len(feature[key]))])
        elif lablel in Slavic:
            y.extend([2 for _ in range(len(feature[key]))])
        elif lablel in Romance:
            y.extend([3 for _ in range(len(feature[key]))])
        elif lablel in others:
            y.extend([4 for _ in range(len(feature[key]))])
        else:
            print(f"Error: {lablel}")
            continue
        X.extend(feature[key])
    return X, y
    

if __name__ == '__main__':
    # Load the data
    #feature = calculate_the_error_feature()
    #feature = get_the_Unigram_feature()
    #feature = get_the_function_words_feature()
    #feature = get_the_sentence_length_feature()
    #feature = get_CharTrigram_Tokens_Unigram_Spelling_Feature()
    feature = get_grammer_spelling_features()
    FEATURE_NAME = 'grammer_spelling'
    KERNEL = 'rbf'
    print(f"Feature {FEATURE_NAME} loaded")
    if DO_WE_NEED_NLI_MODEL:
        
        X, y = create_NLI(feature)
        rus = RandomUnderSampler(random_state=42)
        X, y= rus.fit_resample(X, y)
        X = pd.DataFrame(X)
        X = csr_matrix(X)
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        # Create the model
        model = SVC(kernel=KERNEL)

        # Train the model
        model.fit(X_train, y_train)

        # Test the model
        y_pred_train = model.predict(X_train)
        print("NLI model train accuracy")
        print(accuracy_score(y_train, y_pred_train))

        y_pred = model.predict(X_test)
        print("NLI model")
        print(accuracy_score(y_test, y_pred))
        saving_location = f'{MODELS_LOCATION}\\{FEATURE_NAME}_NLI_model.pkl'

        with open(saving_location, 'wb') as file:
            print("saving the model")
            pickle.dump(model, file)
            print("model saved")
    

    if DO_WE_NEED_FAMILY_MODEL:
        X, y = cereate_family(feature)
        rus = RandomUnderSampler(random_state=42)
        X, y= rus.fit_resample(X, y)
        X = pd.DataFrame(X)
        X = csr_matrix(X)
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        # Create the model
        model = SVC(kernel=KERNEL)

        # Train the model
        model.fit(X_train, y_train)

        # Test the model
        y_pred = model.predict(X_test)

        # Print the accuracy
        print("Family model")
        print(accuracy_score(y_test, y_pred))
        saving_location = f'{MODELS_LOCATION}\\{FEATURE_NAME}_family_model.pkl'
        with open(saving_location, 'wb') as file:
            pickle.dump(model, file)

    if DO_WE_NEED_BINARY_MODEL:
        X, y = create_binary(feature)
        rus = RandomUnderSampler(random_state=42)
        X, y= rus.fit_resample(X, y)
        X = pd.DataFrame(X)
        X = csr_matrix(X)
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        # Create the model
        model = SVC(kernel=KERNEL)

        # Train the model
        model.fit(X_train, y_train)

        # Test the model
        y_pred = model.predict(X_test)

        # Print the accuracy
        print("Binary model")
        print(accuracy_score(y_test, y_pred))
        saving_location = f'{MODELS_LOCATION}\\{FEATURE_NAME}_binary_model.pkl'
        with open(saving_location, 'wb') as file:
            pickle.dump(model, file)