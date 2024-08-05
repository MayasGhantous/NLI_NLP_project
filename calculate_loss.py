import json
import os
import tempfile

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling

from calculate_real_data import read_files_from_directory

# List of languages
languages = ["Turkey", "Slovenia", "Sweden", "Serbia", "Mexico_Spain", "Romania", "Russia", "Poland", "Portugal",
             "Norway", "Lithuania", "Italy", "Hungary", "Greece", "France", "Finland", "Estonia", "Netherlands",
             "Czech", "Croatia", "Bulgaria", "Austria_Germany", "Australia_UK_US_NewZealand_Ireland2"]

DATA_LOCATION = 'europe_data'
fine_tune_location = 'fine_tuning'
output_file = 'evaluation_results2.json'


def create_dataset_from_text(text_list, tokenizer, block_size=128):
    # Join all text samples into a single string
    full_text = "\n".join(text_list)

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
        texts = file.readlines()
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
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=unseen_dataset,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()

    return eval_results


if __name__ == "__main__":
    results = {}
    correct_count = 0

    for unseen_lang in languages:
        validation_file = os.path.join(fine_tune_location, f"{unseen_lang}_validation.txt")
        unseen_texts = load_texts_from_file(validation_file)
        print(f"Loaded {len(unseen_texts)} unseen texts for {unseen_lang}")

        losses = {}
        for model_lang in languages:
            model_dir = os.path.join(fine_tune_location, model_lang)
            eval_results = evaluate_fine_tuned_model(unseen_texts, model_dir)
            losses[model_lang] = eval_results["eval_loss"]
            print(f"Evaluation results for {unseen_lang} on {model_lang} model: {eval_results}")

        min_loss_model = min(losses, key=losses.get)
        min_loss = losses[min_loss_model]
        is_correct_model = min_loss_model == unseen_lang
        results[unseen_lang] = {
            "losses": losses,
            "min_loss_model": min_loss_model,
            "min_loss": min_loss,
            "is_correct_model": is_correct_model
        }
        correct_count += int(is_correct_model)
        print(f"Unseen text: {unseen_lang}, Min loss model: {min_loss_model}, Min loss: {min_loss},"
              f" Is correct model: {is_correct_model}")

    # Save the results to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Calculate and print the accuracy
    total_texts = len(languages)
    accuracy = (correct_count / total_texts) * 100
    print(f"Accuracy: {accuracy:.2f}%")