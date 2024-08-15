# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import optuna
from tqdm.auto import tqdm
from functools import partial
from sklearn.model_selection import StratifiedKFold

# Set random seed for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(42)

# Load the training data
train_date_part = pd.read_csv("C:/Users/vrajc/Downloads/ml_shit/train_date.csv")
train_num_part = pd.read_csv("C:/Users/vrajc/Downloads/ml_shit/train_numeric.csv")

print(train_num_part.shape, train_date_part.shape)

# Plotting the countplot for 'Response'
sns.countplot(x="Response", data=train_num_part)
plt.savefig('countplot_response.png')  # Save the countplot
plt.show()

# Create a pie chart
response_counts = train_num_part['Response'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(response_counts, labels=response_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral'])
plt.title('Pie chart of Response')
plt.savefig('pie_chart_response.png')  # Save the pie chart
plt.show()

# Assuming 'Id' is a continuous variable in train_num_part
plt.figure(figsize=(10, 6))
sns.violinplot(x="Response", y="Id", data=train_num_part)
plt.xlabel('Response')
plt.ylabel('ID')
plt.title('Violin plot of ID by Response')
plt.savefig('violin_plot_id_by_response.png')  # Save the violin plot
plt.show()

# Derive features from train_date_part
stations = train_date_part.count()
indices = stations.reset_index()['index'].str.split('_', expand=True)[1].drop_duplicates().index
train_date_part = train_date_part.iloc[:, indices]

# Create features from train_date.csv
def create_features(df):
    start_station = df.min(axis=1)
    end_station = df.max(axis=1)
    time_diff = end_station - start_station
    return start_station, end_station, time_diff

# Apply the function to the data
train_date_full = train_date_part
start_station, end_station, time_diff = create_features(train_date_full)
train_date_full['start_station'] = start_station
train_date_full['end_station'] = end_station
train_date_full['time_diff'] = time_diff

# Drop rows with all NaN values
train_date_full = train_date_full.dropna(subset=['time_diff'])

# Combine train_numeric and derived features
train_full = train_num_part.merge(train_date_full[['start_station', 'end_station', 'time_diff']], left_index=True, right_index=True)

# Optimize memory usage
def optimize_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            if col_type == 'float64':
                df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                df[col] = df[col].astype('int32')
    return df

train_full = optimize_memory_usage(train_full)

# Handling class imbalance with sampling methods
sampling = 'hybrid'  # Change this as needed

X = train_full.drop(['Response'], axis=1)
y = train_full['Response']

# Handle NaN values
X = X.fillna(-1)  # Fill NaN values with -1 or any suitable value

if sampling == 'under':
    # Implement undersampling
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
elif sampling == 'over':
    # Implement oversampling using SMOTE
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
elif sampling == 'hybrid':
    # Implement hybrid approach
    rus = RandomUnderSampler(sampling_strategy=0.5)
    X, y = rus.fit_resample(X, y)
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Optuna optimization
def optimizer(trial, X, y, K):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    colsample_bynode = trial.suggest_float('colsample_bynode', 0.1, 1.0)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)

    model = XGBClassifier(n_estimators=n_estimators,
                          max_depth=max_depth,
                          colsample_bynode=colsample_bynode,
                          learning_rate=learning_rate,
                          verbosity=0)  # Set verbosity to 0 to suppress detailed output

    folds = StratifiedKFold(n_splits=K)
    scores = []

    for train_idx, val_idx in folds.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]

        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)

# Set the number of K-folds
K = 5

# Create a partial function
opt_func = partial(optimizer, X=X_train, y=y_train, K=K)

# Create a study object and optimize the function
study = optuna.create_study(direction="maximize")
study.optimize(opt_func, n_trials=30)

# Print the best results
print("Best Score (MCC): %.4f" % study.best_value)
print("Best params: ", study.best_trial.params)

# Save Optuna visualizations
# optuna.visualization.plot_optimization_history(study).write_image('optimization_history.png')
# optuna.visualization.plot_param_importances(study).write_image('param_importances.png')

# Retrain the model with the best parameters found
best_params = study.best_trial.params
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Evaluate the model
pred_train = model.predict(X_train)
pred_test = model.predict(X_val)

train_score = f1_score(y_train, pred_train)
test_score = f1_score(y_val, pred_test)

print("Train Score : %.4f" % train_score)
print("Test Score : %.4f" % test_score)

# Calculate MCC score
mcc_train = matthews_corrcoef(y_train, pred_train)
mcc_test = matthews_corrcoef(y_val, pred_test)

print("Train MCC Score : %.4f" % mcc_train)
print("Test MCC Score : %.4f" % mcc_test)

# Load the test data
test_date_part = pd.read_csv("C:/Users/vrajc/Downloads/ml_shit/test_date.csv")
test_num_part = pd.read_csv("C:/Users/vrajc/Downloads/ml_shit/test_numeric.csv")

# Preserve 'Id' column
test_id = test_num_part['Id']

# Prepare test data similarly to the training data
test_date_part = test_date_part.iloc[:, indices]

start_station, end_station, time_diff = create_features(test_date_part)
test_date_part['start_station'] = start_station
test_date_part['end_station'] = end_station
test_date_part['time_diff'] = time_diff
test_date_part = test_date_part.dropna(subset=['time_diff'])

test_full = test_num_part.merge(test_date_part[['start_station', 'end_station', 'time_diff']], left_index=True, right_index=True)
test_full = optimize_memory_usage(test_full)
test_full = test_full.fillna(-1)

# Ensure test_full columns match the trained model's columns
test_full = test_full[X.columns]  # Reorder and align columns to match the training data

# Generate predictions on the full test dataset
pred_test_full = model.predict(test_full)

# Save predictions to CSV with 'Id' and 'Response' columns
output_df_full = pd.DataFrame({
    'Id': test_id,  # Use preserved 'Id'
    'Response': pred_test_full
})
output_df_full.to_csv('solution.csv', index=False, encoding='utf-8')
