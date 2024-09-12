import pickle

class ResultSet:
    def __init__(self, results):
        self.results = results

try:
    with open('/home/ubuntu/icarus/examples/lce-vs-probcache/results.pickle', 'rb') as file:
        loaded_data = pickle.load(file)
        print("Loaded data successfully!")

        # Assuming ResultSet has an attribute 'results'
        results = loaded_data._results
        # print(results)
        with open('/home/ubuntu/icarus/examples/lce-vs-probcache/results.txt', 'w') as txt_file:
            for result in results:
                txt_file.write(str(result) + '\n')

        print("Results written to results.txt successfully!")
except (pickle.PickleError, FileNotFoundError) as e:
    print("Error loading pickle file:", e)

# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt

# # Example dataset loading and preprocessing
# df = pd.read_csv('examples/lce-vs-probcache//events.csv')
# print(df)
# print(pd)
# df = df.iloc[1:500000]  # Subset to the first 499,999 rows

# # Create label for reaccessed data
# df['is_reaccessed'] = df.duplicated(subset='content', keep=False).astype(int)

# # Map priority to numerical values
# df['priority'] = df['priority'].map({'low': 0, 'high': 1})

# # One-hot encode receiver column
# df = pd.get_dummies(df, columns=['receiver'], drop_first=True)

# # Initialize XGBoost model
# xgboost_model = XGBClassifier(eval_metric='logloss')

# # Chunk size and incremental learning loop
# accuracy_history = []
# precision_history = []
# recall_history = []
# f1_history = []

# chunk_size = 10000
# num_chunks = len(df) // chunk_size + 1

# for chunk_number in range(num_chunks):
#     start_idx = chunk_number * chunk_size
#     end_idx = min((chunk_number + 1) * chunk_size, len(df))
#     chunk = df.iloc[start_idx:end_idx]

#     X = chunk.drop(['is_reaccessed', 'timestamp'], axis=1)
#     y = chunk['is_reaccessed']

#     # Adjust test_size and train_size accordingly
#     test_size = 0.3
#     train_size = 1 - test_size

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=20)

#     # Train XGBoost model
#     xgboost_model.fit(X_train, y_train)

#     # Predict on the test set
#     y_pred = xgboost_model.predict(X_test)

#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     # Append metrics to history
#     accuracy_history.append(accuracy)
#     precision_history.append(precision)
#     recall_history.append(recall)
#     f1_history.append(f1)

# # Visualize accuracy trend over chunks
# plt.figure(figsize=(12, 8))

# plt.subplot(2, 2, 1)
# plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o', linestyle='-', color='b')
# plt.xlabel('Chunks')
# plt.ylabel('Accuracy')
# plt.title('Incremental Learning: Model Accuracy over Chunks')
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(range(1, len(precision_history) + 1), precision_history, marker='o', linestyle='-', color='g')
# plt.xlabel('Chunks')
# plt.ylabel('Precision')
# plt.title('Incremental Learning: Model Precision over Chunks')
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(range(1, len(recall_history) + 1), recall_history, marker='o', linestyle='-', color='r')
# plt.xlabel('Chunks')
# plt.ylabel('Recall')
# plt.title('Incremental Learning: Model Recall over Chunks')
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(range(1, len(f1_history) + 1), f1_history, marker='o', linestyle='-', color='purple')
# plt.xlabel('Chunks')
# plt.ylabel('F1-score')
# plt.title('Incremental Learning: Model F1-score over Chunks')
# plt.grid(True)

# plt.tight_layout()
# plt.show()