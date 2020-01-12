from sklearn.datasets import load_boston
import pandas as pd

def boston():
    boston = load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    return boston_df
