import pickle
from icarus.results import ResultSet

input_pickles = [
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/latest_results/results.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/latest_results/results2.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/latest_results/results3.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/latest_results/results4.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/latest_results/results5.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/plots/plots_15/test14/results.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results_aggregated2.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results_aggregated.pickle"
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_21.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_21.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_22.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_23.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_24.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_25.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_26.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_27.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_28.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_29.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_30.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_31.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_32.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_33.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_34.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_35.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_36.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_37.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_38.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_39.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_40.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_41.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_42.pickle",
    
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_43.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_44.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_45.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_49.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_50.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_51.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_52.pickle",
    # "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results/results_aggregated_53.pickle",
    "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results_aggregated.pickle",
    "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results_aggregated3.pickle",
]

merged = ResultSet()

for p in input_pickles:
    with open(p, "rb") as f:
        rs = pickle.load(f)      # rs is a ResultSet
    for cfg, res in rs:          # iterate over (cfg, res) pairs
        merged.add(cfg, res)

with open("/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/results2.pickle", "wb") as f:
    pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
