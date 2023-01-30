
import pandas as pd
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=float, default=0.6)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--target_col', type=str, default='PRICE')
    params, _ = parser.parse_known_args()
    return params


def change_target_to_first_col(df, target_col):
    # shift column 'PRICE' to first position
    first_column = df.pop(target_col)
  
    # insert column using insert(position,column_name,
    # first_column) function
    df.insert(0, target_col, first_column)
    return df


def split_dataset(df, train_size, val_size, test_size, random_state=None):
    """
    Split dataset into train, validation and test samples
    Args:
        df (pandas.DataFrame): input data
        train_size (float): ratio of data to use as training dataset
        val_size (float): ratio of data to use as validation dataset
        test_size (float): ratio of data to use as test dataset
        random_state (int): Pass an int for reproducible output across multiple function calls.
    Returns:
        df_train (pandas.DataFrame): train dataset
        df_val (pandas.DataFrame): validation dataset
        df_test (pandas.DataFrame): test dataset
    """
    if (train_size + val_size + test_size) != 1.0:
        raise ValueError("train_size, val_size and test_size must sum up to 1.0")
    rest_size = 1 - train_size
    df_train, df_rest = train_test_split(
        df,
        test_size=rest_size,
        train_size=train_size,
        random_state=random_state
    )
    df_val, df_test = train_test_split(
        df_rest,
        test_size=(test_size / rest_size),
        train_size=(val_size / rest_size),
        random_state=random_state
    )
    df_train.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    train_perc = int((len(df_train)/len(df)) * 100)
    print(f"Training size: {len(df_train)} - {train_perc}% of total")
    val_perc = int((len(df_val)/len(df)) * 100)
    print(f"Val size: {len(df_val)} - {val_perc}% of total")
    test_perc = int((len(df_test)/len(df)) * 100)
    print(f"Test size: {len(df_test)} - {test_perc}% of total")
    return df_train, df_val, df_test


def scale_dataset(df_train, df_val, df_test, target_col):
    """
    Fit StandardScaler to df_train and apply it to df_val and df_test
    Args:
        df_train (pandas.DataFrame): train dataset
        df_val (pandas.DataFrame): validation dataset
        df_test (pandas.DataFrame): test dataset
        target_col (str): target col
    Returns:
        df_train_transformed (pandas.DataFrame): train data scaled
        df_val_transformed (pandas.DataFrame): val data scaled
        df_test_transformed (pandas.DataFrame): test data scaled
    """
    scaler_data = StandardScaler()
    
    # fit scaler to training dataset
    print("Fitting scaling to training data and transforming dataset...")
    df_train_transformed = pd.DataFrame(
        scaler_data.fit_transform(df_train), 
        columns=df_train.columns
    )
    df_train_transformed[target_col] = df_train[target_col]
    
    # apply scaler to validation and test datasets
    print("Transforming validation and test datasets...")
    df_val_transformed = pd.DataFrame(
        scaler_data.transform(df_val), 
        columns=df_val.columns
    )
    df_val_transformed[target_col] = df_val[target_col]
    df_test_transformed = pd.DataFrame(
        scaler_data.transform(df_test), 
        columns=df_test.columns
    )
    df_test_transformed[target_col] = df_test[target_col]
    return df_train_transformed, df_val_transformed, df_test_transformed


print(f"===========================================================")
print(f"Starting pre-processing")
print(f"Reading parameters")

# reading job parameters
args = read_parameters()
print(f"Parameters read: {args}")

# set input path
input_data_path = "data/raw/house_pricing.csv"

# read data input
df = pd.read_csv(input_data_path)

# move target to first col
df = change_target_to_first_col(df, args.target_col)

# split dataset into train, validation and test
df_train, df_val, df_test = split_dataset(
    df,
    train_size=args.train_size,
    val_size=args.val_size,
    test_size=args.test_size,
    random_state=args.random_state
)

# scale datasets
df_train_transformed, df_val_transformed, df_test_transformed = scale_dataset(
    df_train, 
    df_val, 
    df_test,
    args.target_col
)

df_train_transformed.to_csv('data/script_processed/train/train.csv', sep=',', index=False, header=False)
df_val_transformed.to_csv('data/script_processed/validation/validation.csv', sep=',', index=False, header=False)
df_test_transformed.to_csv('data/script_processed/test/test.csv', sep=',', index=False, header=False)



print(f"Ending pre-processing")
print(f"===========================================================")
