import os
import tempfile

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling

from calculate_real_data import read_files_from_directory

DATA_LOCATION = 'europe_data'
fine_tune_location = 'fine_tuning'


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
    model_path_1 = fine_tune_location + "\\Bulgaria"
    model_path_2 = fine_tune_location + "\\Croatia"

    validation_path_1 = os.path.join(fine_tune_location, "Bulgaria_validation.txt")
    validation_path_2 = os.path.join(fine_tune_location, "Croatia_validation.txt")

    unseen_texts_1 = load_texts_from_file(validation_path_1)

    eval_results = evaluate_fine_tuned_model(unseen_texts_1, model_path_1)
    print("Evaluation results for model's language (Bulgaria):", eval_results)

    unseen_texts_2 = load_texts_from_file(validation_path_2)

    eval_results = evaluate_fine_tuned_model(unseen_texts_1, model_path_2)
    print("Evaluation results for different language (Croatia):", eval_results)