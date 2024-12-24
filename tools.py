from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from huggingface_hub import login
import datasets
from sklearn.model_selection import train_test_split
from datasets import concatenate_datasets
import re
import random

# provides tools for data evaluation
class EvaluateTools:
    def calc_metrics(y_test, predicted_labels):
        accuracy = accuracy_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels)
        recall = recall_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        
        print(f"Model Evaluation Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def plot_metrics(metrics, model_name, dataset_name):
        plt.figure(figsize=(8, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.ylim(0, 1)
        plt.title(f"Model: {model_name}")
        plt.suptitle(f"Dataset: {dataset_name}")
        plt.ylabel("Score")
        plt.xlabel("Metrics")
        for i, v in enumerate(metrics.values()):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        plt.show()

# provides tools for data preprocessing and loading
class DatasetProvider:
    def clean_tweets(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (but keep the text of the hashtag)
        text = re.sub(r'#', '', text)
        # Remove special characters, numbers, and punctuation (except for words)
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def balance_out_dataset(dataset, num_samples):
        # Filter by label
        offense_samples = [sample for sample in dataset if sample['label'] == 1]
        non_offense_samples = [sample for sample in dataset if sample['label'] == 0]
        
            # Ensure balanced sampling
        num_per_class = num_samples // 2
        offense_samples = random.sample(offense_samples, min(num_per_class, len(offense_samples)))
        non_offense_samples = random.sample(non_offense_samples, min(num_per_class, len(non_offense_samples)))

        # Combine and shuffle the samples
        balanced_samples = offense_samples + non_offense_samples
        random.shuffle(balanced_samples)
        
        return datasets.Dataset.from_dict({key: [sample[key] for sample in balanced_samples] for key in balanced_samples[0]})
    
    def get_germ_eval_18(dataset_name="philschmid/germeval18", num_samples=100):
        print("Downloading dataset...")
        data_germeval = datasets.load_dataset(dataset_name)
        data_germeval = concatenate_datasets([
            data_germeval["train"],
            data_germeval["test"]
        ])
        data_germeval = data_germeval.map(lambda x: {'text': DatasetProvider.clean_tweets(x['text']), 'label': (1 if x['binary'] == "OFFENSE" else 0)})
        data_germeval = data_germeval.remove_columns(['binary', 'multi'])
        return DatasetProvider.balance_out_dataset(data_germeval, num_samples)

    def get_superset(dataset_name = "manueltonneau/german-hate-speech-superset", num_samples=100):
        load_dotenv()
        # Ensure that the Hugging Face authentication token is available in the .env file
        token = os.getenv("HF_AUTH_TOKEN")
        if token is None:
            raise ValueError("Hugging Face authentication token not found in .env file.")
        login(token=token)

        print("Downloading dataset...")
        data_superset = datasets.load_dataset(dataset_name, split="train")
        data_superset = data_superset.map(lambda x: {'text': DatasetProvider.clean_tweets(x['text']), 'label': x['labels']})
        data_superset = data_superset.remove_columns(['labels', 'source', 'dataset', 'nb_annotators'])    
        # balancing out the data does not work as well for the superset as it does only have a fairly small amount of positive samples
        return DatasetProvider.balance_out_dataset(data_superset, num_samples)

    def stats(dataset):
        print("Sample Data:")
        print(dataset[0])
        print("\nData Length:")
        print(len(dataset))
        labels = [example['label'] for example in dataset]
        print(f"Number of labels with 1: {labels.count(1)}")
        print(f"Number of labels with 0: {labels.count(0)}")

    def split_data(data, test_size=0.2):
        texts = []
        labels = []
        for sample in data:
            found_text = ""
            found_label = ""
            try:
                found_text = sample['text']
                found_label = int(sample['label'])
            except:
                continue
            texts.append(found_text)
            labels.append(found_label)

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test