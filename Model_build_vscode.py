# Imports

import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer


import warnings
 
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn import metrics

from sklearn.pipeline import Pipeline
 
# Models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

import joblib

warnings.filterwarnings("ignore")

# Data import
data = pd.read_csv("cleaned_data.csv")
print(data.head())

# data backup
df = data.copy()

# explore data
df.info()

# Encode target column
df["Churn"] = df["Churn"].map({'Yes':1, 'No':0})
print(df["Churn"].value_counts())

# Separate target and feature
X = df.drop(columns= ["Churn"], axis =1)
y = df["Churn"]

num_columns = X.select_dtypes(include= "number").columns.to_list()
cat_columns = X.select_dtypes(include= "object").columns.to_list()

print("\n THE NUMERIC COLUMNS ARE: \n", num_columns)
print("\n THE CATEGORICAL COLUMNS ARE: \n", cat_columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Processing data
num_transformer = Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy= "most_frequent")),
                           ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# Combine steps
preprocessor = ColumnTransformer(transformers = [("num",num_transformer,num_columns),
                                               ("cat",cat_transformer,cat_columns)])


# Models
model_dict = {"Logistic_Regression": LogisticRegression(max_iter=1000),
             "SVM": SVC(probability=True),
             "Random_Forest": RandomForestClassifier(),
             "Adaboost": AdaBoostClassifier(),
             "XgBoost": XGBClassifier(use_label_encoder = False, eval_metric = "logloss")}

# Hyperparameter
# Generic hyperparameter
search_space = {
    "C": [0.1,1,10],
    "kernel": ["linear", "rgf"],
    "n_estimators": [50,100,200],
    "max_depth": [None,5,10],
    "learning_rate" : [0.5,1]
}

# Function to filter hyperparameter
def filter_hyperparameters(model,space):
    valid_key = model.get_params().keys()
    return {k:v for k,v in space.items() if k in valid_key}

# Grid Search for each model

result = []

best_pipeline = {}
 
for name, model in model_dict.items():

    print(f'Tuning {model}...')

    pipe = Pipeline(steps = [

        ('processor', preprocessor),

        ('model', model)

    ])

    hyperparameter = filter_hyperparameters(model, search_space)

    # Prefix model name

    param_grid = {f'model__{k}':v for k,v in hyperparameter.items()}

    grid = GridSearchCV(estimator = pipe, param_grid = param_grid, 

                        cv = 5, scoring = 'accuracy', n_jobs = -1)

    grid.fit(X_train, y_train)
 
    y_pred = grid.predict(X_test)

    report = metrics.classification_report(y_test, y_pred, output_dict = True)
 
    result.append({

        'model_name': name,

        'best_parameters': param_grid,

        'accuracy': round(metrics.accuracy_score(y_test, y_pred), 4),

        'f1-score': round(report['weighted avg']['f1-score'], 4)

    })
 
    best_pipeline[name] = grid.best_estimator_

print(result)

# Compare results
result_df =pd.DataFrame(result)
sorted_result_df= result_df.sort_values(by= "accuracy", ascending = False)
print("\nModel comparision:\n", sorted_result_df)

# Best Model
best_row = sorted_result_df.iloc[0]
best_model = best_row["model_name"]
print("\nBest Model:",best_model)
print("\nBest hyperparameters: \n", best_row["best_parameters"])

# Retraining best model on full data set
final_pipeline= best_pipeline[best_model]
final_pipeline.fit(X,y)

# Save pipeline
joblib.dump(final_pipeline, "churn_pipeline_2.pkl")
print("The deployment model is saved as: churn_pipeline_2.pkl")