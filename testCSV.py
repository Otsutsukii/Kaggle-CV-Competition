import pandas as pd 

df = pd.DataFrame()
df["Id"]=pd.Series(list(range(100)))
df["Category"] = pd.Series(list(range(99)))
df.to_csv("submission.csv",index=False)