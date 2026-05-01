import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1) Load data
df = pd.read_csv("summer_sales_dataset.csv")

feature_cols = df.columns[:-1]
label_col = df.columns[-1]

X = df[feature_cols]
y = df[label_col]

# 2) Train logistic regression
clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
clf.fit(X, y)

# 3) Predict
df["predicted_product"] = clf.predict(X)

print(df[feature_cols.tolist() + [label_col, "predicted_product"]].head(20))

