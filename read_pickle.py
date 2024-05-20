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
        print(results)
except (pickle.PickleError, FileNotFoundError) as e:
    print("Error loading pickle file:", e)
