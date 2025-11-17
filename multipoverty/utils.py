import datetime
import pandas as pd

def rescale(X,scaler):
    return scaler.fit_transform(X)

def read_dataset(path: str):
    df = pd.read_csv(path,index_col=0).iloc[:,1:]
    return df.values

def get_timestamp():
    return datetime.datetime.now().strftime('%H:%M:%S')