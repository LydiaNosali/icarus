import pandas as pd
# Specify the column names for the unlabeled data
column_names = ['data_back', 'timestamp', 'name', 'size', 'priority', 'interest_life_time', 'response_time']

file_name = 'icarus/models/strategy/traces/IBMObjectStoreTrace000Part0-0.2.csv'

# Read the original trace data with specified column names
trace_data = pd.read_csv(file_name, names=column_names)

# Select and rename the required columns
transformed_data = trace_data[['timestamp', 'name', 'size', 'priority']]


# # Rename the columns
transformed_data.rename(columns={'name': 'content'}, inplace=True)

# # Map the priority values to the desired format if needed
priority_mapping = {'l': 'low', 'h': 'high'}
transformed_data['priority'] = transformed_data['priority'].map(priority_mapping)

# Save the transformed data to a new CSV file
transformed_data.to_csv(file_name, index=False)

# Print a message indicating the file has been saved
print("Transformed data has been saved to" + file_name)
