import os
import tempfile
from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from calculate_real_data import read_files_from_directory


DATA_LOCATION = 'europe_data'
fine_tune_location = 'fine_tuning'


def merge_files_for_country(files_data):
    merged_texts = []
    for key, texts in files_data.items():
        current_country, person = key
        merged_texts.extend(texts)
    return merged_texts

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

def fine_tune_gpt2(train_dataset, test_dataset, output_dir):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model()


def save_text_to_file(text_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(text_list))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('device: ', device)


    # Update countries_list to have lists of country IDs
    countries_list = [
        [20],  # romania
        [18],  # poland
        [15],  # netherland
        [14, 24],   # mexico and spain
        [9],   # greece
        [1, 8],  # austria and germany
        [0, 27, 28, 16, 11]     # Australia,UK,US,NewZealand and Ireland
    ]
    countries_names = ['Romania', 'Poland', 'Netherlands', 'Mexico_Spain', 'Greece',
                       'Austria_Germany', 'Australia_UK_US_NewZealand_Ireland']  # Update names accordingly
    
    index = 0
    for country_ids, country_name in zip(countries_list, countries_names):
        files_data = defaultdict(list)

        for country_id in country_ids:
            files_data.update(read_files_from_directory(DATA_LOCATION, country_id))

        country_location = os.path.join(fine_tune_location, country_name)
        validation_location = os.path.join(fine_tune_location, f"{country_name}_validation.txt")

        merged_texts = merge_files_for_country(files_data)

        if not merged_texts:
            raise ValueError(
                "No text data found for the specified country. Ensure the data is correctly read and processed.")

        print(f"Number of text samples: {len(merged_texts)}")

        # Split the data into 80% train and 20% temp set
        train_texts, temp_texts = train_test_split(merged_texts, test_size=0.2, random_state=42)

        # Split the temp set into 50% test and 50% validation (10% each of original data)
        test_texts, validation_texts = train_test_split(temp_texts, test_size=0.5, random_state=42)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        train_dataset = create_dataset_from_text(train_texts, tokenizer)
        test_dataset = create_dataset_from_text(test_texts, tokenizer)

        # Save the validation texts to a file
        save_text_to_file(validation_texts, validation_location)

        print(f"Fine-tuning GPT-2 for {countries_names[index]}...")
        fine_tune_gpt2(train_dataset, test_dataset, country_location)
        print("Fine-tuning complete. Model saved.")
        index += 1
