import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
max_itr = 5

for i in range(1, max_itr + 1):
    sampling_path = os.path.join(base_dir, f"final_sampling_itr_{i}.csv")
    drop_path = os.path.join(base_dir, f"final_drop_itr_{i}.csv")
    output_path = os.path.join(base_dir, f"final_itr_{i}.csv")

    sampling_df = pd.read_csv(sampling_path)
    drop_df = pd.read_csv(drop_path)

    result_df = pd.concat([sampling_df, drop_df], ignore_index=True)
    result_df.to_csv(output_path, index=False)

    print(f"itr_{i}: sampling={len(sampling_df)}, drop={len(drop_df)}, final={len(result_df)}")
