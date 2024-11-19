from typing import List

import pandas as pd


def create_unique_identifier(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """
    Create a unique identifier for each row in a dataframe based on the values of the
    columns specified in the list.

    Example usage:
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    create_unique_identifier(df, ['a', 'b'])
    returns: 0    1__4
             1    2__5
             2    3__6

    :param df: pandas dataframe
    :param columns: list of column names
    :return: pandas series with unique identifier
    """
    series = None
    for col in columns:
        if series is None:
            series = df[col].astype(str)
        else:
            series = series + "__" + df[col].astype(str)
    return series
