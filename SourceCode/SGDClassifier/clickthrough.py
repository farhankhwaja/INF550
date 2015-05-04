import sklearn
import pandas as pd

csv_file = "E:\\INF 550 Dataset\\train.csv"
df = pd.read_csv(csv_file, index_col=0)
print df