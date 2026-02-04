import pandas as pd
from pdb import set_trace

# Data with repeated numbers
data = [1,1,1,1,2,2,2,2,2,3,3,3,7,8,8,8]

# Divide into quartiles
quantile_labels = pd.qcut(data, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
set_trace()

print(quantile_labels)

