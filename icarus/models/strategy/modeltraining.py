import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# List all trace files in the directory
traces_directory = "examples/lce-vs-probcache/traces"
trace_files = [f for f in os.listdir(traces_directory) if f.endswith('.csv')]

trace_names = []
# Initialize lists to store metrics
accuracy_history = []
precision_history = []
recall_history = []
f1_history = []

for filename in trace_files:
    file_path = os.path.join(traces_directory, filename)
    df = pd.read_csv(file_path, names=['timestamp', 'receiver', 'content', 'size', 'priority'])
    
    # Perform preprocessing steps on the individual trace
    df = df.iloc[1:500000]  # Subset to the first 499,999 rows
    # Create label for reaccessed data
    df['is_reaccessed'] = df.duplicated(subset='content', keep=False).astype(int)

    # Map priority to numerical values
    df['priority'] = df['priority'].map({'low': 0, 'high': 1})

    # Convert 'content' column to string to ensure uniform encoding
    df['content'] = df['content'].astype(str)

    df['timestamp'] = df['timestamp'].astype(float)
    
    
    # Encode 'content' column with label encoding
        # Encode 'content' and 'receiver' columns with label encoding
    label_encoder_content = LabelEncoder()
    # label_encoder_receiver = LabelEncoder()
    df['content'] = label_encoder_content.fit_transform(df['content'])
    # df['receiver'] = label_encoder_receiver.fit_transform(df['receiver'])
    
    # Convert 'size' column to numeric (float or int)
    df['size'] = pd.to_numeric(df['size'], errors='coerce')

    # Calculate inter-arrival time 
    df['prev_timestamp'] = df['timestamp'].shift(1) 
    df['inter_arrival_time'] = df['timestamp'] - df['prev_timestamp']

    df['inter_arrival_time'].fillna(0, inplace=True)
    
    # Previous access count and time since last access 
    df['prev_access_count'] = df.groupby('content').cumcount() 
    df['time_since_last_access'] = df.groupby('content')['timestamp'].diff().fillna(0) 
    
    # Select relevant features for modeling 
    X = df.drop(['receiver','is_reaccessed', 'timestamp', 'prev_timestamp'], axis=1) 
    # X = df.drop(['is_reaccessed', 'timestamp'], axis=1) 
    y = df['is_reaccessed'] 
    
    # Split the data into training and test sets
    test_size = 0.3
    train_size = 1 - test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=20)

    # Train the XGBoost model
    xgboost_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgboost_model.predict(X_test)

    trace_names.append(filename[:4])

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics to history
    accuracy_history.append(accuracy)
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)

# Print the metrics
print("Accuracy history: ", accuracy_history)
print("Precision history: ", precision_history)
print("Recall history: ", recall_history)
print("F1-score history: ", f1_history)

# Optionally, you can also visualize the metrics
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(trace_names, accuracy_history, marker='o', linestyle='-', color='b')
plt.xlabel('Traces')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Traces')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(trace_names, precision_history, marker='o', linestyle='-', color='g')
plt.xlabel('Traces')
plt.ylabel('Precision')
plt.title('Model Precision over Traces')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(trace_names, recall_history, marker='o', linestyle='-', color='r')
plt.xlabel('Traces')
plt.ylabel('Recall')
plt.title('Model Recall over Traces')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(trace_names, f1_history, marker='o', linestyle='-', color='purple')
plt.xlabel('Traces')
plt.ylabel('F1-score')
plt.title('Model F1-score over Traces')
plt.grid(True)

plt.tight_layout()
plt.show()

