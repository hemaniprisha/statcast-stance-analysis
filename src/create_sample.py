import pandas as pd

df = pd.read_csv("../data/combined_statcast_data.csv")

# Old-style, works on all versions, still triggers a FutureWarning
sample_df = df.groupby("player_name", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 100))
)

sample_df.to_csv("../data/sample_statcast.csv", index=False)
