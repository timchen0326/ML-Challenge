import pandas as pd

# === Config ===
RAW_FILE = "data/original/cleaned_data_combined_modified.csv"
TEST_FILE = "test_raw.csv"
LABEL_FILE = "true_labels.txt"
N_SAMPLES = 100

# === Load and sample ===
df = pd.read_csv(RAW_FILE)
df_sampled = df.sample(n=N_SAMPLES)

# # === Save raw test file ===
# df_sampled.to_csv(TEST_FILE, index=False)
# print(f"Test file saved: {TEST_FILE} ({len(df_sampled)} rows)")

# # === Optionally save true labels ===
# if "Label" in df_sampled.columns:
#     df_sampled["Label"].to_csv(LABEL_FILE, index=False, header=False)
#     print(f"True labels saved: {LABEL_FILE}")

# === Save raw test file WITHOUT the label ===
df_sampled.drop(columns=["Label"]).to_csv(TEST_FILE, index=False)
print(f"Test file saved: {TEST_FILE} ({len(df_sampled)} rows)")

# === Save true labels separately for local eval ===
df_sampled["Label"].to_csv(LABEL_FILE, index=False, header=False)
print(f"True labels saved: {LABEL_FILE}")
