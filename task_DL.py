# -*- coding: utf-8 -*-
"""Task_DL.ipynb
Original file is located at
    https://colab.research.google.com/drive/1gAnvdFHADX5wDzQ9kx_n9Q4A8BZTDWuD
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

# Load dataset without header
df = pd.read_csv("/content/drive/MyDrive/TasK/train_data (1).csv",
                 header=None, names=["text", "rate"])

df

# Delete the row containing 0 and 1 label
df=df.drop(index=0).reset_index(drop=True)

import re

# removing html tag using regular expression
df['text']=df['text'].astype(str).apply(lambda x: re.sub(r'<.*?>', '', x))

# after removing Html tags
print(df.head(5))

# Show full cell content (no truncation)
pd.set_option('display.max_colwidth', None)

# Print first 5 rows
print(df.head(5))

# Removing punctuation
df['text']=df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print(df.head(5))

# Removing Numbers from text column using number regular expression
df['text']=df['text'].apply(lambda x: re.sub(r'\d+', '', x))
print(df.head(5))

# Converting Lowercase
df['text']=df['text'].str.lower()
print(df.head(5))

# Using NLTK for mitigating Stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# removing stopwords
df['text']=df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

print(df.head(5))

df

#importing scikitlearn
from sklearn.model_selection import train_test_split

# Spliting train as 80% and Test as 20%
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['rate'],
    test_size=0.2,
    random_state=42,
    stratify=df['rate']
)


print("Train Data size:", X_train.shape[0])
print("Test Data size:", X_test.shape[0])









######### ------------- OUR BERT model Start HERE ------------------#######

!pip install transformers datasets torch -q
from datasets import Dataset

train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_dataset  = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

from transformers import AutoTokenizer # importing tokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# setting pytorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=[],  # Explicitly disable W&B logging
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train() # training

# Saving bert trained model with our custom dataset
import joblib
joblib.dump(model, "bert_model.pkl")

##### Evaluationg the model
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predicting test data
y_pred = trainer.predict(test_dataset).predictions

# converting class label
y_pred = np.argmax(y_pred, axis=1)

# actual labels
y_true = np.array(y_test)

#  Report is generating
print(" Classification Report for BERT:")
print(classification_report(y_true, y_pred, digits=4))

# confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues)
plt.title("BERT Confusion Matrix")
plt.show()

############### ------------------- OUR BERT model ENDHs HERE ---------------########








######### ----------------- Our RoBERT Model Start HERE --------------------------- ###########

from datasets import Dataset

train_dataset = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_dataset  = Dataset.from_dict({"text": X_test.tolist(),  "label": y_test.tolist()})

#  TOkeninzing text uisng Robert tokenizer
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset  = test_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Loading Robert model here
from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

training_args = TrainingArguments(
    output_dir="./roberta_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=[],  # disabling
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


# Saving bert trained model with our custom dataset
import joblib
joblib.dump(model, "Ro_bert_model.pkl")


######## Evaluating the model 

import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predicting test data
y_pred = trainer.predict(test_dataset).predictions

# converting class label
y_pred = np.argmax(y_pred, axis=1)

# actual labels
y_true = np.array(y_test)

#  Report is generating
print(" Classification Report for RoBERT:")
print(classification_report(y_true, y_pred, digits=4))

# confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues)
plt.title("RoBERT Confusion Matrix")
plt.show()

######### ----------------- OUR RoBERT model ENds HERE ----------------#########










######### ----------Ensemble Method integraing Bert and RoBert with CatBoost meta learner  Model Start HERE ------------------- #########

!pip install transformers catboost
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading Bert
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device) # model

# loading roBert
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModel.from_pretrained("roberta-base").to(device)   # model

# Function to get CLS token embeddings with batching
def get_cls_embeddings(texts, tokenizer, model, batch_size=16): # Reduced batch size to 16
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        # CLS token is at position 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

# Extracting features from test and train data

# bert embeddings
bert_train = get_cls_embeddings(X_train, bert_tokenizer, bert_model)
bert_test = get_cls_embeddings(X_test, bert_tokenizer, bert_model)

# RoBert embeddings
roberta_train = get_cls_embeddings(X_train, roberta_tokenizer, roberta_model)
roberta_test = get_cls_embeddings(X_test, roberta_tokenizer, roberta_model)

# adding features
X_train_combined = np.hstack([bert_train, roberta_train])
X_test_combined = np.hstack([bert_test, roberta_test])

# model training
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

cat_model = CatBoostClassifier(
    iterations=500,  # we used 200 too
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    verbose=100
)

cat_model.fit(X_train_combined, y_train)


# ----------- Save the model as .pkl -----------
import joblib
joblib.dump(cat_model, "bert_roberta_catboostEnsemble_500.pkl")

from sklearn.metrics import  ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predictions
y_pred = cat_model.predict(X_test_combined)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
plt.title("Confusion Matrix - BERT + RoBERTa + CatBoost_500ite")
plt.show()

########### --------------------- OUR ensmeble Model ENDs HERE ------------------ ###########









