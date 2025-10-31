import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load train/test splits
train_df = pd.read_csv(r"D:\FakeNewsDetection\data\processed\bert_train_combined.csv")
test_df = pd.read_csv(r"D:\FakeNewsDetection\data\processed\bert_test_combined.csv")


# Use only the first 512 tokens per article (BERT max length)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_batch(df):
    return tokenizer(
        list(df['combined_text']),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

train_encodings = encode_batch(train_df)
test_encodings = encode_batch(test_df)

# Convert labels to tensors
train_labels = torch.tensor(train_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",        # <--- already "epoch"
    save_strategy="epoch",              # <--- add this line!
    logging_dir='./logs',
    logging_steps=20,
    save_total_limit=1,
    load_best_model_at_end=True,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Training BERT... (this may take some time on CPU)")
trainer.train()

# Evaluate
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(-1)
y_true = test_df['label'].values
print("Test Accuracy:", accuracy_score(y_true, y_pred))
