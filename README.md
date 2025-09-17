# Austin House Price Range Prediction 🏡

This project aims to predict the price range of houses in Austin, TX, using various property attributes. The problem is framed as a multi-class classification task, where we classify each property into one of five predefined price brackets.

The repository showcases a complete machine learning workflow, including exploratory data analysis, feature engineering, model training, evaluation, and the incorporation of Natural Language Processing (NLP) features from property descriptions.

---

## 📋 Project Pipeline

The project follows these key steps:

1.  **Exploratory Data Analysis (EDA):**
    * Analyzed numerical and categorical features to understand their distributions and relationships with the target variable (`priceRange`).
    * Visualized correlations between features, revealing strong multicollinearity between `MedianStudentsPerTeacher`, `numOfBedrooms`, and `numOfBathrooms`.

2.  **Feature Engineering:**
    * Created a new feature, `ageBuilding`, from the `yearBuilt` column to represent the age of a property.
    * Dropped redundant or highly correlated features to improve model performance and reduce noise.

3.  **Model Training & Evaluation:**
    * **Baseline Model:** A **Logistic Regression** model was trained to establish a performance baseline.
    * **Primary Model:** A **LightGBM Classifier** was implemented, which significantly outperformed the baseline.
    * Models were evaluated using **Weighted F1-Score** and **Accuracy** due to the multi-class nature of the problem.

4.  **Feature Importance:**
    * Analyzed feature importances from the trained LightGBM model to identify the key drivers of house price ranges. **Location (`latitude`, `longitude`)**, **property age (`ageBuilding`)**, and **school quality (`avgSchoolRating`)** were identified as the most influential factors.

5.  **NLP Feature Integration:**
    * Developed a text processing pipeline using **TF-IDF Vectorization** on the `description` column to extract valuable information and further enhance the model's predictive power. The text is cleaned by removing stop words and applying lemmatization.

---

## 📊 Results

The LightGBM model demonstrated superior performance compared to the Logistic Regression baseline.

| Model                        | F1-Score (Weighted) | Accuracy |
| :--------------------------- | :------------------ | :------- |
| Logistic Regression (Baseline) | 0.47                | 0.47     |
| **LightGBM Classifier** | **0.62** | **0.62** |

### Key Predictive Features

The feature importance analysis highlighted that a property's price range is most influenced by:
1.  **Geographical Location** (`latitude` & `longitude`)
2.  **Property Age** (`ageBuilding`)
3.  **Average School Rating** (`avgSchoolRating`)
4.  **Lot Size** (`lotSizeSqFt`)

![Feature Importance Plot](https://i.imgur.com/L53R0cW.png)

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. The required libraries are listed in `requirements.txt`.
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
nltk
joblib

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/glopezzz/ML_challenge_FS.git](https://github.com/glopezzz/ML_challenge_FS.git)
    cd ML_challenge_FS
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download NLTK data (required for NLP feature processing):
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### Usage

1.  **Run the Analysis:** Open and run the `live_exercise.ipynb` notebook in a Jupyter environment to see the full analysis, training, and evaluation process.

2.  **Make Predictions:** The trained model pipeline is saved in the `/outputs` directory. You can use the `predict_price_range` function to predict on new data.

    ```python
    import pandas as pd
    import joblib

    # Load the trained model and label encoder
    model_pipeline = joblib.load('./outputs/lgbm_price_range_pipline.joblib')
    label_encoder = joblib.load('./outputs/label_encoder.joblib')

    # Create a sample DataFrame with new data (ensure columns match the training data)
    new_data = pd.DataFrame({
        'city': ['austin'],
        'homeType': ['Single Family'],
        'latitude': [30.26],
        'longitude': [-97.74],
        'garageSpaces': [2],
        'hasSpa': [True],
        'numOfPatioAndPorchFeatures': [1],
        'lotSizeSqFt': [8000.0],
        'avgSchoolRating': [8.5],
        'numOfBathrooms': [3.0],
        'ageBuilding': [15]
        # 'description' would be needed for the final NLP model
    })

    # Predict
    prediction_encoded = model_pipeline.predict(new_data)
    predicted_price_range = label_encoder.inverse_transform(prediction_encoded)
    probabilities = model_pipeline.predict_proba(new_data)

    print(f"Predicted Price Range: {predicted_price_range[0]}")
    print(f"Probabilities: {probabilities}")
    ```

---

## 🏛️ Repository Structure
.
├── data/
│   └── train.csv           # Training dataset
├── outputs/
│   ├── lgbm_price_range_pipline.joblib  # Saved model pipeline
│   └── label_encoder.joblib           # Saved label encoder
├── .gitignore
├── live_exercise.ipynb     # Main Jupyter Notebook with all analysis
└── README.md

## 🔮 Future Work

The notebook outlines several next steps to further improve the model:
* **Full NLP Integration:** Complete the training and evaluation of the model using the TF-IDF features from the property descriptions.
* **Hyperparameter Tuning:** Use `GridSearchCV` or a similar technique to optimize the hyperparameters of the LightGBM classifier and the TF-IDF Vectorizer.
* **Model Deployment:** Package the final model into a simple API (e.g., using Flask or FastAPI) for real-time predictions.

