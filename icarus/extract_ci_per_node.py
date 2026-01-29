import os
import pandas as pd

def load_nodes_ci(folder, day_str, period):
    """
    Load 24h carbon intensity for all node CSV files in 'folder'
    for a given date 'day_str' (e.g. '2024-03-10').

    Returns:
        nodes_ci: dict[node_id] -> list of 24 values (hour 0..23)
    """
    nodes_ci = {}
    target_date = pd.to_datetime(day_str).date()

    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(folder, fname)
        df = pd.read_csv(path)
        if "Datetime (UTC)" not in df.columns:
            raise ValueError(f"{fname} missing 'Datetime (UTC)' column")

        df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
        df["date"] = df["Datetime (UTC)"].dt.date

        df_day = df[df["date"] == target_date].copy()
        if df_day.empty:
            raise ValueError(f"No data for date {day_str} in file {fname}")

        # Ensure sorted and take first 24 rows
        df_day = df_day.sort_values("Datetime (UTC)").iloc[:24]

        if "Carbon intensity gCO₂eq/kWh (direct)" not in df_day.columns:
            raise ValueError(f"{fname} missing direct carbon intensity column")

        df_day["hour"] = df_day["Datetime (UTC)"].dt.hour
        # This assumes exactly one entry per hour
        ci_by_hour = df_day.set_index("hour")["Carbon intensity gCO₂eq/kWh (direct)"]

        # Fill list of length 24; if an hour is missing, you can choose a default
        ci = []
        for h in range(period):
            if h in ci_by_hour.index:
                ci.append(float(ci_by_hour.loc[h]))
            else:
                # Fallback: repeat previous or use overall mean
                ci.append(float(ci_by_hour.mean()))

        nodes_ci[df["Zone name"][0]] = ci

    return nodes_ci

# folder = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/carbon_intensities/carbon_profiles/"  # path containing all node CSVs
# day_str = "2024-05-24"       # choose a representative day
# nodes_ci = load_nodes_cih(folder, day_str)
# print(f"node_ci:{nodes_ci}")
# print(nodes_ci["ES"][0])
