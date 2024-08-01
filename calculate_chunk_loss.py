import json
import os
import tempfile

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling
import tqdm
from calculate_real_data import read_files_from_directory
import warnings
import io
import sys
# Suppress specific warning (FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# List of languages
languages = ["Turkey", "Slovenia", "Sweden", "Serbia", "Mexico_Spain", "Romania", "Russia", "Poland", "Portugal",
             "Norway", "Lithuania", "Italy", "Hungary", "Greece", "France", "Finland", "Estonia", "Netherlands",
             "Czech", "Croatia", "Bulgaria", "Austria_Germany", "Australia_UK_US_NewZealand_Ireland"]

DATA_LOCATION = 'europe_data'
fine_tune_location = 'fine_tuning'
output_file = 'evaluation_chunk_results.json'


def create_dataset_from_text(text_list, tokenizer, block_size=128):
    # Join all text samples into a single string
    #full_text = "\n".join(text_list)
    full_text = text_list

    # Create a temporary file to store the text
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(full_text)
        temp_file_path = temp_file.name

    # Create dataset from the temporary file
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=temp_file_path,
        block_size=block_size,
    )

    # Remove the temporary file
    os.unlink(temp_file_path)

    return dataset


def load_texts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.read()
    return texts


def evaluate_fine_tuned_model(unseen_texts, model_dir):
    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Prepare the unseen dataset
    unseen_dataset = create_dataset_from_text(unseen_texts, tokenizer)

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Set up evaluation arguments
    training_args = TrainingArguments(
        per_device_eval_batch_size=8,
        output_dir='./results',  # You can specify any directory here
        disable_tqdm=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=unseen_dataset,
    )

    # Evaluate the model
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    eval_results = trainer.evaluate()
    sys.stdout = old_stdout
    return eval_results
def create_list(text1):
    list_of_texts = []
    texts = text1.split("\n\n")
    for text in texts:
        list_of_texts.append(text)
    return list_of_texts


if __name__ == "__main__":
    results = {}
    correct_count = 0
    over_all_counter = 0
    for unseen_lang in languages:
        
        validation_file = os.path.join(fine_tune_location, f"{unseen_lang}_validation.txt")
        unseen_texts = load_texts_from_file(validation_file)
        list_of_unseen_texts = create_list(unseen_texts)
        
        print(f"Loaded {len(list_of_unseen_texts)} unseen texts for {unseen_lang}")

        current_corrent = 0 
        current_results = []
        for i,text in enumerate(tqdm.tqdm(list_of_unseen_texts)):
            losses = {}
            for model_lang in languages:
                #print(f"Calculating loss for {model_lang} model")
                model_dir = os.path.join(fine_tune_location, model_lang)
                eval_results = evaluate_fine_tuned_model(text, model_dir)
                losses[model_lang] = eval_results["eval_loss"]
            min_loss_model = min(losses, key=losses.get)
            
            if  min_loss_model==unseen_lang:
                #print(f'Correct prediction for {unseen_lang} model index {i}')
                current_corrent += 1
                correct_count += 1
            current_results.append(min_loss_model)  
            over_all_counter += 1
        print(f"Unseen text: {unseen_lang} accuracy: {current_corrent/len(list_of_unseen_texts)*100}")


            
        print(f"Evaluation results for {unseen_lang} on {model_lang} model: {eval_results}")

        min_loss = losses[min_loss_model]
        is_correct_model = min_loss_model == unseen_lang
        results[unseen_lang] = {
            "accuracy": current_corrent/len(list_of_unseen_texts)*100,
            "results": current_results,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    # Save the results to a file
    results["overall_accuracy"] = correct_count / over_all_counter * 100
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Calculate and print the accuracy
    total_texts = len(languages)
    accuracy = (correct_count / total_texts) * 100
    print(f"Accuracy: {accuracy:.2f}%")
