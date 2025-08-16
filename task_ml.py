# -*- coding: utf-8 -*-
"""Task_ML.ipynb
Original file is located at
    https://colab.research.google.com/drive/1DoVEU9BHjFnabmSi8nhAw8msG49muX_c
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

print(df.head(5))  # printing

df                  # printing

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



# imporing tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_df=0.95,          # ignore very common words
    min_df=2,
    sublinear_tf=True
)

# Fit on training data and transform both train and test
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)

print("TF-IDF feature matrix shapes:")
print("Train:", X_train_tfidf.shape)
print("Test :", X_test_tfidf.shape)





### (1) ----------------------- Logistic Regression Start HERE ------------ ##

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Initialize Logistic Regression
lr_model = LogisticRegression(
    max_iter=2000,        # ensure convergence
    solver='liblinear',   # good for sparse data
    class_weight='balanced',
    random_state=42
)

# Train the model
lr_model.fit(X_train_tfidf, y_train)

import joblib
# Save the trained Logistic Regression model
joblib.dump(lr_model, '/content/drive/MyDrive/TasK/logistic_regression_model.pkl')

from sklearn.metrics import accuracy_score, classification_report
# Predict on test set
y_pred = lr_model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = ['0', '1']  # 0 = Negative, 1 = Positive

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Labels
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix  Logistic Regression'
)

# Annotate counts in the squares
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()

####-----------   Logistic Regression End  HERE ----------#####




####- (2)  -----------  Naive Bayes Code Start  HERE ----------------###

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Initialize Multinomial Naive Bayes
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train_tfidf, y_train)

import joblib
# Save the trained Logistic Regression model
joblib.dump(lr_model, '/content/drive/MyDrive/TasK/naive_bayes.pkl')

y_pred  = nb_model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = ['0', '1']  # 0 = Negative, 1 = Positive

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Labels
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix  Naive Bayes'
)

# Annotate counts in the squares
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()

##### --------- Naive Bayes model ends ----------#




#### ---------------  Just applied a Grid Search CV (start)  ----------#
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

# Define hyperparameter grid
param_grid = {'alpha': [0.1,0.2, 0.5, 1.0, 1.5, 2.0]}

# Grid search with 5-fold CV
grid_nb = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_nb.fit(X_train_tfidf, y_train)

print("Best alpha:", grid_nb.best_params_)
print("Best CV Accuracy:", grid_nb.best_score_)

#### ----------------- Just applied a Grid Search CV ends ----------------- #






####  (3)------------------------ Using KNN models start ---------------##

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Initialize KNN
knn_model = KNeighborsClassifier(
    n_neighbors=5,      # n value or neighbour value we are assuming 5
    metric='cosine',    # using cosine simillarity here
    n_jobs=-1
)

# Train the model
knn_model.fit(X_train_tfidf, y_train)

# Prediction on Test data
y_pred  = knn_model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = ['0', '1']  # 0 = Negative, 1 = Positive

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Labels
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix  KNN '
)

# Annotate counts in the squares
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()
########## ----------------- KNN model Ends --------------- ######





#####  (4)--------------- Using KNN with Keras Hyper parameter Tuning to select n_neighbours  Starts HERE--------- ##

!pip install keras-tuner -q   # installing keras tunar 

from kerastuner import HyperModel, RandomSearch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Split training data further for tuning
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# HyperModel defination for KNN
class KNNHyperModel(HyperModel):
    def build(self, hp):
        n_neighbors = hp.Int('n_neighbors', 1, 20, step=1)
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='cosine',
            n_jobs=-1
        )
        return model

# KerasTuner search
class KNNWrapper(HyperModel):
    def build(self, hp):
        return KNeighborsClassifier(
            n_neighbors=hp.Int('n_neighbors', 1, 20, step=1),
            metric='cosine',
            n_jobs=-1
        )

# Using Randomsearch Tunar
import kerastuner as kt

best_acc = 0
best_k = 5

#  Searching the Best n_neighbours for KNN model
for k in range(1, 21):
    knn_model = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn_model.fit(X_train_sub, y_train_sub)
    y_val_pred = knn_model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"n_neighbors={k}, Validation Accuracy={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest n_neighbors={best_k} with Validation Accuracy={best_acc:.4f}")

# Train final KNN with best n_neighbors
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric='cosine', n_jobs=-1)
knn_model.fit(X_train_tfidf, y_train)

import joblib
# Save the trained KNN model
joblib.dump(knn_model, '/content/drive/MyDrive/TasK/Keras_Tuned_knn_model.pkl')

# Evaluate on test set
y_pred = knn_model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Predict on test set
y_test_pred = knn_model.predict(X_test_tfidf)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
classes = ['0', '1']  # 0 = Negative, 1 = Positive

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Set labels
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='True label',
    xlabel='Predicted label',
    title=f'Confusion Matrix — KNN (n_neighbors={knn_model.n_neighbors})'
)

# Annotate counts in squares
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()

######  --------- Hyper parameter tuned KNN Ends here --------------#####






####  (5)------- Using Ensemble Technique Integrating the Logistic Regression, Naive Bayes, KNN and CatBoost as meta-learner  Start HERE ----###

!pip install catboost -q         # Installing CaBoost 

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


### base Models

log_reg = LogisticRegression(max_iter=500, n_jobs=-1)
nb = MultinomialNB()
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)

### CatBoost Mete learner here
cat_meta = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0
)


### Stacking Ensemble with CatBoost meta learner
stack_model = StackingClassifier(
    estimators=[
        ('logreg', log_reg),
        ('nb', nb),
        ('knn', knn)
    ],
    final_estimator=cat_meta,
    n_jobs=-1
)

### Training Stacking ensemble model
stack_model.fit(X_train_tfidf, y_train)

import joblib
joblib.dump(stack_model, "stacking_catboost_model.pkl")

# Evaluating Ensemble model
y_pred = stack_model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

### Confusion Matrix for Stacking ensemble model

cm = confusion_matrix(y_test, y_pred)
classes = ['0', '1']

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix — Stacking Ensemble(CatBoost)'
)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.show()

#######------------- Stacking Ensemble model ends here ---------- ####





##### ------ Displaying Training Loss and Training Accuracy with Validation too for Stacking Ensemble model*--------###

from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import numpy as np

# Prepare train and validation data for CatBoost
train_pool = Pool(X_train_tfidf, y_train)
val_pool = Pool(X_test_tfidf, y_test)

# Train CatBoost with logging
catboost_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=10
)

catboost_model.fit(train_pool, eval_set=val_pool, plot=False)

# Get metrics
metrics = catboost_model.get_evals_result()

# Plot Training Loss
plt.figure(figsize=(12,5))
plt.plot(metrics['learn']['MultiClass'], label='Training Loss (Logloss)')
plt.plot(metrics['validation']['MultiClass'], label='Validation Loss (Logloss)')
plt.xlabel("Iteration")
plt.ylabel("Logloss")
plt.title("CatBoost Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Accuracy
plt.figure(figsize=(12,5))
plt.plot(metrics['learn']['Accuracy'], label='Training Accuracy')
plt.plot(metrics['validation']['Accuracy'], label='Validation Accuracy')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("CatBoost Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

###### ---------------- ENDS here ------------ ####






#### (6)------------ Using Decision Tree Classifier Model With Pre pruning technique to handle Overfitting tendency  Start Here ------ ####

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


dt_model = DecisionTreeClassifier(
    criterion='entropy',       # used entropy
    max_depth=20,              # tree depth
    min_samples_split=10,      # at least sample need to split
    min_samples_leaf=5,        # minimum sample in leaf
    random_state=42
)

# model training
dt_model.fit(X_train_tfidf, y_train)

import joblib
# ---- Save the model ----
joblib.dump(dt_model, "decision_tree_model.pkl")

# prediction
y_pred = dt_model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(
    dt_model,
    X_test_tfidf,
    y_test,
    cmap=plt.cm.Blues
)
plt.title("Decision Tree Confusion Matrix (Pre-pruned)")
plt.show()

#######---------------------  Decision Tree Ends Here --------------- ##########







###### -------------- USing SVM classifer --------------------- #########

# ----------------- Importing  -----------------
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------- Initialize SVM classifier-----------------
svm_model = SVC(
    kernel='linear',      # linear kernel works well for text
    C=1.0,                # regularization parameter
    probability=True,     # allows probability estimates
    random_state=42
)

# ----------------- Train SVM -----------------
svm_model.fit(X_train_tfidf, y_train)


# ----------------- Predict -----------------
y_pred = svm_model.predict(X_test_tfidf)

# ----------------- Classification Report -----------------
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------- Confusion Matrix -----------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - SVM Classifier")
plt.show()

#####  -------------------- SVM classifier Ends Here ---------------------- ########

