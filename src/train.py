# ---------------------------------------------
# Import required libraries
# ---------------------------------------------

import pandas as pd                      # Data handling
import joblib                            # Model saving
from datasets import load_dataset        # Load from Hugging Face

from sklearn.model_selection import train_test_split, GridSearchCV  # Splitting + tuning
from sklearn.ensemble import RandomForestClassifier                 # Model
from sklearn.metrics import accuracy_score, classification_report   # Evaluation


# ---------------------------------------------
# Step 1: Load Dataset from Hugging Face
# ---------------------------------------------

def load_data():
    # Load dataset from Hugging Face hub
    dataset = load_dataset("karthickshiva07/engine-predictive-maintenance")

    # Convert to pandas dataframe
    df = dataset["train"].to_pandas()

    # Clean column names (remove spaces issues)
    df.columns = df.columns.str.strip()

    # Rename columns for consistency
    df.rename(columns={
        "Engine rpm": "Engine_RPM",
        "Lub oil pressure": "Lub_Oil_Pressure",
        "Fuel pressure": "Fuel_Pressure",
        "Coolant pressure": "Coolant_Pressure",
        "lub oil temp": "Lub_Oil_Temperature",
        "Coolant temp": "Coolant_Temperature",
        "Engine Condition": "Engine_Condition"
    }, inplace=True)

    return df


# ---------------------------------------------
# Step 2: Data Preparation
# ---------------------------------------------

def preprocess_data(df):

    # Separate features and target
    X = df.drop("Engine_Condition", axis=1)
    y = df["Engine_Condition"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------
# Step 3: Model Training + Hyperparameter Tuning
# ---------------------------------------------

def train_model(X_train, y_train):

    # Define base model
    model = RandomForestClassifier(random_state=42)

    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    # GridSearch for tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    # Train model
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Print best parameters
    print("Best Parameters:", grid_search.best_params_)

    return best_model


# ---------------------------------------------
# Step 4: Model Evaluation
# ---------------------------------------------

def evaluate_model(model, X_test, y_test):

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ---------------------------------------------
# Step 5: Save Model
# ---------------------------------------------

def save_model(model):

    # Save trained model
    joblib.dump(model, "model/engine_failure_model.pkl")

    print("\nModel saved successfully!")

# ---------------------------------------------
# Main Execution
# ---------------------------------------------

def main():

    # Load data
    df = load_data()

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

# Run script
if __name__ == "__main__":
    main()