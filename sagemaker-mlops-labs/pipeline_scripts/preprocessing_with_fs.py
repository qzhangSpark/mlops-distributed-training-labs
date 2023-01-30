import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sagemaker'])

from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd
import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import sagemaker
import boto3


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
    parser.add_argument('--train_feature_group_name', type=str, default='fs-train')
    parser.add_argument('--validation_feature_group_name', type=str, default='fs-validation')
    parser.add_argument('--test_feature_group_name', type=str, default='fs-test')
    parser.add_argument('--bucket_prefix', type=str, default='mlops-workshop/feature-store')
    parser.add_argument('--target_col', type=str, default='PRICE')
    parser.add_argument('--region', type=str)
    parser.add_argument('--role_arn', type=str)
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


def prepare_df_for_feature_store(df, data_type):
    """
    Add event time and record id to df in order to store it in SageMaker Feature Store
    Args:
        df (pandas.DataFrame): data to be prepared
        data_type (str): train/validation or test
    Returns:
        df (pandas.DataFrame): dataframe with event time and record id
    """
    print(f'Preparing {data_type} data for Feature Store..')
    current_time_sec = int(round(time.time()))
    # create event time
    df['event_time'] = pd.Series([current_time_sec]*len(df), dtype="float64")
    # create record id from index
    df['record_id'] = df.reset_index().index
    return df
    

def wait_for_feature_group_creation_complete(feature_group):
    """
    Function that waits for feature group to be created in SageMaker Feature Store
    Args:
        feature_group (sagemaker.feature_store.feature_group.FeatureGroup): Feature Group
    """
    status = feature_group.describe().get('FeatureGroupStatus')
    print(f'Initial status: {status}')
    while status == 'Creating':
        print(f'Waiting for feature group: {feature_group.name} to be created ...')
        time.sleep(5)
        status = feature_group.describe().get('FeatureGroupStatus')
    if status != 'Created':
        raise SystemExit(f'Failed to create feature group {feature_group.name}: {status}')
    print(f'FeatureGroup {feature_group.name} was successfully created.')
    
# a Feature Group is the main Feature Store resource, it is a logical grouping of features
def create_feature_group(feature_group_name, sagemaker_session, df, prefix, role_arn):
    """
    Create Feature Store Group
    Args:
        feature_group_name (str): Feature Store Group Name
        sagemaker_session (sagemaker.session.Session): sagemaker session
        df (pandas.DataFrame): dataframe to injest used to create features definition
        prefix (str): geature group prefix (train/validation or test)
        role_arn (str): role arn to create feature store
    Returns:
        fs_group (sagemaker.feature_store.feature_group.FeatureGroup): Feature Group
    """
    fs_group = FeatureGroup(
        name=feature_group_name, 
        sagemaker_session=sagemaker_session
    )
    fs_group.load_feature_definitions(data_frame=df)
    default_bucket = sagemaker_session.default_bucket()
    print(f'Creating feature group: {fs_group.name} ...')
    fs_group.create(
        s3_uri=f's3://{default_bucket}/{prefix}', 
        record_identifier_name='record_id', 
        event_time_feature_name='event_time', 
        role_arn=role_arn, 
        enable_online_store=True
    )
    wait_for_feature_group_creation_complete(fs_group)
    return fs_group


def ingest_features(fs_group, df):
    """
    Ingest features to Feature Store Group
    Args:
        fs_group (sagemaker.feature_store.feature_group.FeatureGroup): Feature Group
        df (pandas.DataFrame): dataframe to injest
    """
    print(f'Ingesting data into feature group: {fs_group.name} ...')
    fs_group.ingest(data_frame=df, max_processes=3, wait=True)
    print(f'{len(df)} records ingested into feature group: {fs_group.name}')
    return


print(f"===========================================================")
print(f"Starting pre-processing")
print(f"Reading parameters")

# reading job parameters
args = read_parameters()
print(f"Parameters read: {args}")
sagemaker_session = sagemaker.Session(boto3.Session(region_name=args.region))

# set input path
input_data_path = "/opt/ml/processing/input/house_pricing.csv"

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

# prepare datasets for Feature Store
df_train_transformed_fs = prepare_df_for_feature_store(df_train_transformed, 'train')
df_val_transformed_fs = prepare_df_for_feature_store(df_val_transformed, 'validation')
df_test_transformed_fs = prepare_df_for_feature_store(df_test_transformed, 'test')

# injest datasets to Feature Store
fs_group_train = create_feature_group(
    args.train_feature_group_name, 
    sagemaker_session, 
    df_train_transformed_fs, 
    args.bucket_prefix+'/train',
    args.role_arn
)
ingest_features(fs_group_train, df_train_transformed_fs)

fs_group_validation = create_feature_group(
    args.validation_feature_group_name, 
    sagemaker_session, 
    df_val_transformed_fs, 
    args.bucket_prefix+'/validation',
    args.role_arn
)
ingest_features(fs_group_validation, df_val_transformed_fs)

fs_group_test = create_feature_group(
    args.test_feature_group_name, 
    sagemaker_session, 
    df_test_transformed_fs, 
    args.bucket_prefix+'/test',
    args.role_arn
)
ingest_features(fs_group_test, df_test_transformed_fs)

print(f"Ending pre-processing")
print(f"===========================================================")
