# pyrefly: ignore [missing-import]
from pandas import plotting
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLE'] = 'True'
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available?: {torch.cuda.is_available()}")

model_name = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Num labels 6 because we have 6 CEFR levels (A1, A2, B1, B2, C1, C2)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

df = pd.read_csv("data/processed/merged_data.csv")

def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True,        
        max_length=512,         
        padding="max_length"    
    )

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=31)
    return train_df, test_df

def prepare_data(df):
    train_df, test_df = split_data(df)
    train_dataset = Dataset.from_pandas(train_df[["text", "label_encode"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label_encode"]])

    train_dataset = train_dataset.rename_column("label_encode", "labels")
    test_dataset = test_dataset.rename_column("label_encode", "labels")

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    return tokenized_train, tokenized_test


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # CEFR levels can be unbalanced, so 'weighted' f1 is better
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
    }


def train_model(tokenized_train, tokenized_test):
    # LoRA Config
    lora_config = LoraConfig(
        r=16,                # Matrix size (keeping it low saves memory)
        lora_alpha=32,
        target_modules=["query_proj", "value_proj"], # Layers for DeBERTa model
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS # For sequence classification task
    )

    # Get PEFT model
    model_lora = get_peft_model(model, lora_config)

    model_lora.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir='./deberta_cefr_optimized',
        
        # --- Epoch ve Batch Config ---
        num_train_epochs=7,
        per_device_train_batch_size=8,   
        gradient_accumulation_steps=4,   
        
        # --- Learning Rate Config ---
        learning_rate=1e-4,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.15,
        
        # --- Regularization Config ---
        weight_decay=0.05,
        label_smoothing_factor=0.1,
        
        # --- Save and Evaluation Config ---
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Best model is selected based on f1 score
        
        # --- Hardware Config ---
        fp16=False,
        bf16=True,
        average_tokens_across_devices=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    return trainer,trainer.model

def plot_confusion_matrix(trainer, test_df):

    save_path = "results/confusion_matrix.png"

    # Create directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    predictions = trainer.predict(test_df)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    cefr_classes = ["A1", "A2", "B1", "B2", "C1", "C2"]

    # Create matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=cefr_classes, yticklabels=cefr_classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CEFR Level Classification Complexity Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def save_model(model):
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained("models/cefr_deberta_lora")
    tokenizer.save_pretrained("models/cefr_deberta_lora")


if __name__ == "__main__":
    tokenized_train, tokenized_test = prepare_data(df)
    trainer, trained_model = train_model(tokenized_train, tokenized_test)
    plot_confusion_matrix(trainer, tokenized_test)
    save_model(trained_model)





