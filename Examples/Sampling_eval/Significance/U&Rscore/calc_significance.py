"""
Significance test for Score vs Score(Expert).

Score = 60 + 40/(Gold - Rand) * (Exp - Rand)

Since Gold and Exp are fixed and only Rand is random, by error propagation:
  dScore/dRand = 40 * (Exp - Gold) / (Gold - Rand)^2
  Std(Score) = |dScore/dRand| * Std(Rand)

For Score(Expert), Exp_expert is recovered from the given mean:
  Score(Expert) = 60 + 40*(Exp_expert - Rand)/(Gold - Rand)
  => Exp_expert = Rand + (Score_expert - 60)*(Gold - Rand)/40

Paired t-test (n=5 runs, paired because same Rand in each run):
  D = Score - Score(Expert)
  Std(D) = |f'(Rand) - g'(Rand)| * Std(Rand)
         = |40*(Exp - Exp_expert)/(Gold - Rand)^2| * Std(Rand)
  t = mean(D) / (Std(D)/sqrt(n)), df = n-1 = 4
"""

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("literature_test.csv")
print(df.to_string(), "\n")

n = 5  # number of repeated runs

results = []
for _, row in df.iterrows():
    Gold = row["Gold"]
    Rand = row["Rand"]
    RandStd = row["Rand Std"]
    Exp = row["Exp"]
    Score = row["Score"]
    Score_expert = row["Score (Expert)"]

    denom = Gold - Rand  # (Gold - Rand)

    # --- Std of Score via error propagation ---
    dScore_dRand = 40 * (Exp - Gold) / denom**2
    std_score = abs(dScore_dRand) * RandStd

    # --- Recover Exp_expert from mean Score(Expert) ---
    # Score_expert = 60 + 40*(Exp_expert - Rand)/(Gold - Rand)
    Exp_expert = Rand + (Score_expert - 60) * denom / 40

    # --- Std of Score(Expert) via error propagation ---
    dScoreExp_dRand = 40 * (Exp_expert - Gold) / denom**2
    std_score_expert = abs(dScoreExp_dRand) * RandStd

    # --- Paired t-test ---
    # D_i = Score_i - Score_expert_i  (same Rand_i in each run)
    # Std(D) = |f'(Rand) - g'(Rand)| * Std(Rand)
    diff_derivative = dScore_dRand - dScoreExp_dRand  # = 40*(Exp-Exp_expert)/denom^2
    std_diff = abs(diff_derivative) * RandStd

    D_mean = Score - Score_expert
    se_diff = std_diff / np.sqrt(n)
    t_stat = D_mean / se_diff
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)  # two-tailed, df=4

    results.append({
        "Dimension": row["Dimension"],
        "Substrates": row["Substrates"],
        "Score": Score,
        "Std(Score)": round(std_score, 4),
        "Score(Expert)": Score_expert,
        "Std(Score Expert)": round(std_score_expert, 4),
        "D_mean": round(D_mean, 4),
        "Std(D)": round(std_diff, 4),
        "t": round(t_stat, 4),
        "p": round(p_value, 4),
        "Significant (p<0.05)": p_value < 0.05,
    })

res = pd.DataFrame(results)
print(res.to_string(index=False))
