import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data_path = './data/'
df = pd.read_csv(data_path+'train.csv')
df.head()
df.info()
df.drop('uid', axis=1, inplace=True)
# Label balance analysis. Multi class/label
print(df['priceRange'].value_counts()/len(df))

df['buildingAge'] = 2025 - df['yearBuilt']
print(df[['buildingAge','yearBuilt']])
df.drop('yearBuilt', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['priceRange_encoded'] = le.fit_transform(df['priceRange'])
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

df.drop(['MedianStudentsPerTeacher', 'numOfBedrooms'], axis=1, inplace=True)

# Categorical Feature analysis
print(df['city'].value_counts())

plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
sns.countplot(data=df, y='homeType', hue='priceRange')
plt.show()
plt.subplot(1,2,2)
sns.countplot(data=df, x='hasSpa', hue='priceRange')

plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='priceRange', 
                palette='viridis', alpha=0.6, s=10)
plt.title('Geospatial Distribution of Properties by Price Range in Austin')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Price Range')
plt.show()

# Data prep before training
X = df.drop(['description', 'priceRange', 'priceRange_encoded'], axis=1)
y = df['priceRange']
y_encoded = le.fit_transform(y)

numerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','bool']).columns.tolist()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
 
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(max_iter=1000))])

lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)

f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# LightGBM
import lightgbm as lgb
lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', lgb.LGBMClassifier(importance_type='gain'))])
"""
>'Split' Importance: This is the most straightforward metric. 
It simply counts how many times a feature was used to make a split in
 a decision tree across the entire model. Think of it as a frequency counter. 
 A high 'split' importance means a feature was frequently used by the model to make decisions.
>'Gain' Importance: This metric is more focused on performance. 
It measures the total improvement in the model's accuracy (or the total reduction in its error)
 that is achieved by using a particular feature for splitting. 
 Every time a feature is used to split a node, the model calculates
 how much that split improved the objective function;
 'gain' is the sum of all those improvements for that feature across all trees. 
 It essentially quantifies the predictive contribution of a feature."""

lgbm_pipeline.fit(X_train, y_train)

y_pred_lgbm = lgbm_pipeline.predict(X_test)

f1_lgbm = f1_score(y_test, y_pred_lgbm, average='weighted')
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)

# Compare model results
print('Logistic Regression Results:')
print('F1-Score: ',f1_lr)
print('Accuracy: ',accuracy_lr)

print('LigthGBM Results:')
print('F1-Score: ',f1_lgbm)
print('Accuracy: ',accuracy_lgbm)

# Analyze feature importance from LightGBM
feature_names = list(numerical_cols) + \
                list(lgbm_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_cols))
feature_importances = lgbm_pipeline.named_steps['classifier'].feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importances from LightGBM')
plt.show()


import joblib

output_dir = './output_models/'
# Save target encoder
label_encoder_filename = 'label_encoder.joblib'
joblib.dump(le, output_dir + label_encoder_filename)

model_filename = 'lgbm_price_range_model.joblib'
joblib.dump(lgbm_pipeline, output_dir + model_filename)


# Function to score new data points
def predict_price_range(data: pd.DataFrame):
    
    output_dir = './output_models/'

    try:
        model_pipeline = joblib.load(output_dir + 'lgbm_price_range_model.joblib')
        label_encoder = joblib.load(output_dir + 'label_encoder.joblib')
    except:
        "Error: Label Encoder or Model not found"

    prediction_encoded = model_pipeline.predict(data)
    predicted_price_range = label_encoder.inverse_transform(prediction_encoded)

    probabilities = model_pipeline.predict_proba(data)
    probabilities = {label_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}

    return predicted_price_range, probabilities

# Test the funtion:
row_to_test = 42
sample = X.iloc[row_to_test]
# sample = X.iloc[row_to_test:row_to_test+1]
print(sample)

predicted_range, probs = predict_price_range(sample)

print('Predicted Price Range: ', predicted_range)
print('Probabilities: ', probs.items())
print('Real Price Range: ', df.iloc[row_to_test]['priceRange'])

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the tools that are working
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemma_tokenizer(text):
    if not isinstance(text, str):
        return []
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. TOKENIZATION (The Fix)
    # Use Python's fast, built-in split() method.
    # This completely avoids the NLTK tokenizer bug.
    tokens = text.split()

    # 4. & 5. Stop-Word removal and Lemmatization
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 4]

    return processed_tokens


# --- Testing the function ---
test_desc = df.iloc[0]['description']
print("Original Description:")
print(test_desc)
print("\nPreprocessed Description:")
print(lemma_tokenizer(test_desc))

from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize the TF-IDF Vectorizer
# We will use our custom preprocessing function
# max_features limits the vocabulary to the top 2000 terms
# ngram_range=(1, 2) captures both single words and two-word phrases
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=lemma_tokenizer,
    max_features=2000,
    ngram_range=(1, 2)
)


# Define the final preprocessor that includes the TF-IDF vectorizer
final_preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('nlp', tfidf_vectorizer, 'description')
    ],
    remainder='passthrough'
)

final_pipeline = Pipeline(
    steps=[
        ('preprocessor', final_preprocessor),
        ('classifier', lgb.LGBMClassifier())
        ]
    )

X_full = df.drop(['priceRange','priceRange_encoded','description_length'], axis=1)
X_full.fillna('',inplace=True)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_encoded, test_size = 0.2, random_state=42, stratify=y_encoded
)

final_pipeline.fit(X_train_full, y_train_full)

y_pred_full = final_pipeline.predict(X_test_full)

f1_final = f1_score(y_pred=y_pred_full, y_true=y_test_full, average='weighted')
acc_final = accuracy_score(y_pred=y_pred_full, y_true=y_test_full)

# Compare model results
print('Logistic Regression Results:')
print('F1-Score: ',f1_lr)
print('Accuracy: ',accuracy_lr)

print('LigthGBM Results:')
print('F1-Score: ',f1_lgbm)
print('Accuracy: ',accuracy_lgbm)

print('LigthGBM + NLP Results:')
print('F1-Score: ',f1_final)
print('Accuracy: ',acc_final)

# Improvements:

print()