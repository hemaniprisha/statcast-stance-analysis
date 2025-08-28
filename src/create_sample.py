# Creates sample dataset from the combined_statcast_data.csv file
import pandas as pd

df = pd.read_csv("../data/combined_statcast_data.csv")

sample_df = df.groupby("player_name", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 100))
)

sample_df.to_csv("../data/sample_statcast.csv", index=False)
