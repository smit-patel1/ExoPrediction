import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

exo_df = pd.read_csv("../data/exoplanets.csv")

# Selecting the features
selected_cols = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_model_snr", "koi_impact", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

exo_df_modeling = exo_df[selected_cols + ["koi_disposition"]]

# Dropping the empty rows
exo_df_processed = exo_df_modeling.dropna()

exo_df_processed = exo_df_processed.copy()

# Encoding the target variable
mapping = {
    "CONFIRMED": 0,
    "CANDIDATE": 1,
    "FALSE POSITIVE": 2
}

exo_df_processed["koi_disposition_encoded"] = exo_df_processed["koi_disposition"].map(mapping)

# Spliting into train and test
X = exo_df_processed[selected_cols]

y = exo_df_processed["koi_disposition_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


exo_df_processed.to_csv("../data/exoplanets_processed.csv", index=False)


