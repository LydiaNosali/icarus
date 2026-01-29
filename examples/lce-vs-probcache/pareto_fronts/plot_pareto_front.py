import os
import pandas as pd
import matplotlib.pyplot as plt

N_PERIOD = 24
plotdir = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/pareto_fronts"
topsis_path = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/pareto_fronts/exp1_TELEKOM_topsis.csv"

os.makedirs(plotdir, exist_ok=True)
records = []

for period in range(N_PERIOD):
    df = pd.read_csv(f"{plotdir}/exp1_TELEKOM_p{period}.csv")
    
    df_topsis = pd.read_csv(topsis_path)
    topsis_point = df_topsis[df_topsis["period"] == period].iloc[0]
    
    # stats sur le front Pareto de ce period
    hit_min   = df["hit"].min()
    hit_max   = df["hit"].max()
    hit_mean  = df["hit"].mean()
    hit_std   = df["hit"].std()          # par défaut, std « échantillon » (n-1)

    cost_min  = df["cost"].min()
    cost_max  = df["cost"].max()
    cost_mean = df["cost"].mean()
    cost_std  = df["cost"].std()

    carbon_min  = df["carbon"].min()
    carbon_max  = df["carbon"].max()
    carbon_mean = df["carbon"].mean()
    carbon_std  = df["carbon"].std()

    max_hit = df.loc[df["hit"].idxmax()]
    min_cost = df.loc[df["cost"].idxmin()]
    min_carbon = df.loc[df["carbon"].idxmin()]
    
    records.append({
        "period": period,

        "max_hit_hit": max_hit["hit"],
        "max_hit_cost": max_hit["cost"],
        "max_hit_carbon": max_hit["carbon"],

        "min_cost_hit": min_cost["hit"],
        "min_cost_cost": min_cost["cost"],
        "min_cost_carbon": min_cost["carbon"],

        "min_carbon_hit": min_carbon["hit"],
        "min_carbon_cost": min_carbon["cost"],
        "min_carbon_carbon": min_carbon["carbon"],
        # nouvelles stats
        "hit_min": hit_min,
        "hit_max": hit_max,
        "hit_mean": hit_mean,
        "hit_std": hit_std,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "cost_mean": cost_mean,
        "cost_std": cost_std,
        "carbon_min": carbon_min,
        "carbon_max": carbon_max,
        "carbon_mean": carbon_mean,
        "carbon_std": carbon_std,
    })


    # ---------- Hit vs Carbon ----------
    plt.figure()
    plt.scatter(df["carbon"], df["hit"], alpha=0.6, label="Pareto solutions")
    plt.scatter(
        topsis_point["carbon"],
        topsis_point["hit"],
        color="red",
        marker="X",
        s=120,
        label="TOPSIS"
    )
    plt.xlabel("Carbon footprint")
    plt.ylabel("Cache hit")
    plt.title(f"Pareto front + TOPSIS (CI init) - period {period}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(plotdir, f"cf_h_ci_only_pareto_period_{period}.png"),
        bbox_inches="tight"
    )
    plt.close()

    # ---------- Cost vs Carbon ----------
    plt.figure()
    plt.scatter(df["carbon"], df["cost"], alpha=0.6, label="Pareto solutions")
    plt.scatter(
        topsis_point["carbon"],
        topsis_point["cost"],
        color="red",
        marker="X",
        s=120,
        label="TOPSIS"
    )
    plt.xlabel("Carbon footprint")
    plt.ylabel("Cost")
    plt.title(f"Pareto front + TOPSIS (CI init) - period {period}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(plotdir, f"cf_c_ci_only_pareto_period_{period}.png"),
        bbox_inches="tight"
    )
    plt.close()

    # ---------- Hit vs Cost ----------
    plt.figure()
    plt.scatter(df["cost"], df["hit"],  alpha=0.6, label="Pareto solutions")
    plt.scatter(
        topsis_point["carbon"],
        topsis_point["hit"],
        color="red",
        marker="X",
        s=120,
        label="TOPSIS"
    )
    plt.xlabel("Cost")
    plt.ylabel("Cache hit")
    plt.title(f"Pareto front + TOPSIS (CI init) - period {period}")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(plotdir, f"c_h_ci_only_pareto_period_{period}.png"),
        bbox_inches="tight"
    )
    plt.close()


extreme_df = pd.DataFrame(records)
extreme_df.to_csv("/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/pareto_fronts_ci_only/extreme_solutions_ci_only.csv", index=False)

plt.figure()
# plt.plot(extreme_df["period"], extreme_df["hit_std"], label="std(hit)")
# plt.plot(extreme_df["period"], extreme_df["cost_std"], label="std(cost)")
plt.plot(extreme_df["period"], extreme_df["carbon_std"], label="std(carbon)")
plt.xlabel("Period")
plt.ylabel("Std dev")
plt.legend()
plt.grid(True)
plt.show()

# plt.figure()
# plt.plot(extreme_df["period"], extreme_df["max_hit_carbon"], label="Max Hit", linestyle="--")
# plt.plot(extreme_df["period"], extreme_df["min_carbon_carbon"], label="Min Carbon", linestyle="--")
# plt.plot(extreme_df["period"], extreme_df["min_cost_carbon"], label="Min Cost", linestyle="--")
# plt.plot(df_topsis["period"], df_topsis["carbon"], marker="X", color="red", label="TOPSIS")
# plt.xlabel("Period")
# plt.ylabel("Carbon footprint")
# plt.title("Extremes + TOPSIS over time")
# plt.grid(True)
# plt.legend()
# plt.show()