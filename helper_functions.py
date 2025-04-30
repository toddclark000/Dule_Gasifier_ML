import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime
from datetime import timedelta


####################################################### general purpose data frame wrangling help ###################################################################3 
def to_datetime(df):
    # Create a mask for rows that contain AM or PM in the Time column
    mask_ampm = df['Time'].str.contains('AM|PM')

    # Initialize a new column for the datetime results
    df['Datetime'] = pd.NaT

    # Process rows with AM/PM
    df.loc[mask_ampm, 'Datetime'] = pd.to_datetime(
        df.loc[mask_ampm, 'Date'] + ' ' + df.loc[mask_ampm, 'Time'],
        format='%m/%d/%Y %I:%M:%S %p'
    )

    # Process rows without AM/PM (military time)
    df.loc[~mask_ampm, 'Datetime'] = pd.to_datetime(
        df.loc[~mask_ampm, 'Date'] + ' ' + df.loc[~mask_ampm, 'Time'],
        format='%m/%d/%Y %H:%M:%S'
    )

    df = df.drop(columns = ['Date', 'Time'])
    
    return df


def plot_all_fields(df, title):
    '''
    Plots each non-time column in a compact 3-column grid of subplots vs. the 'Datetime' column.
    Subplot titles have extra space, x-labels are hidden, and y-ticks use smaller font.
    '''
    time_column = 'Datetime'
    fields = [col for col in df.columns if col != time_column]
    num_fields = len(fields)

    ncols = 3
    nrows = math.ceil(num_fields / ncols)
    height_per_row = 3  # very compact

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15, height_per_row * nrows),
        sharex=False
    )

    axes = axes.flatten()

    for i, field in enumerate(fields):
        ax = axes[i]
        ax.plot(df[time_column], df[field])
        ax.set_title(field, fontsize=8, pad=8)  # pad=8 adds vertical space above plot
        ax.tick_params(labelbottom=False, labelsize=7)  # remove x labels, shrink y ticks

    # Hide unused axes
    for j in range(len(fields), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.75, wspace=0.4)  # More vertical space for titles
    plt.show()




def get_relevant_fields(df):

    column_filter = [
        'Date', 'Time',
        #   'EFG Air Flow Rate lb/hr', 'Primary O2 Flow Rate lb/hr',
        #             'EFG System Pressure psi',
        #             'Slurry Flow Rate lb/hr',
        #             'EFG:EFG_Oxygen_Pressure_1.Value',
        #             'EFG:EFG_Oxygen_Pressure_2.Value',
                    ###############################################################################
                    # 'Analyzer 2 CO2 %',                 # Is it worth keeping these extra analyzers?
                    # #'Analyzer 2 CO %',broken           # Tf they are just being kept because they
                    # 'Analyzer 3 O2 %',                  # have errors than that is exactly what we
                    # 'Analyzer 3 CO2%',                  # want.
                    # 'ABB CH4',
                    # 'ABB CO',
                    # 'ABB CO2',
                    # 'ABB O2',
                    ###############################################################################
                    # 'GB O2%',
                    # 'GB CO2%',
                    'GB CO%',
                    # 'GB CH4%',
                    'GB H2%',
                    # 'GB C2H6%',
                    # 'GB C2H2%',
                    # 'GB LHV [MJ/Nm3]',
                    'EFG:B_TC_1_Shallow.Value'
                    # 'EFG:B_TC_4_Shallow.Value'
                    ]
    
    if 'Moyno Flow Rate lb/hr' in df.columns:
        df.rename(columns={'Moyno Flow Rate lb/hr': 'Slurry Flow Rate lb/hr'}, inplace=True)
    
    return df[column_filter]




####################################################################### test functions ############################################################################
def test_nulls_and_zeros(df, field_name:str):
    null_count = df[field_name].isnull().sum()
    zero_count = (df[field_name] == 0).sum()
    return {'null count': null_count, 'zero count': zero_count}

def rmse(ground_truth_df, df_to_test, window_start:datetime=None, window_end:datetime=None ):
    ''' 
    1. Checks if all fields and times match up, Throws error if they dont
    2. Calculates RMSE over time for all fields except "Datetime"
    3. Returns dict with fields as keys and RMSE as values
    '''
    # check if fields match
    if not ground_truth_df.columns.equals(df_to_test.columns):
        raise ValueError("DataFRames have different columns")
    
    # check that dates line up
    if not ground_truth_df["Datetime"].equals(df_to_test["Datetime"]):
        raise ValueError("Dates do not match")
    
    # Apply window if given
    if window_start is not None and window_end is not None:
        mask = (ground_truth_df["Datetime"] >= window_start) & (ground_truth_df["Datetime"] <= window_end)
        ground_truth_df = ground_truth_df[mask].copy()
        df_to_test = df_to_test[mask].copy()
    
    # calculate RMSE
    rmse_dict = {}
    for col in ground_truth_df.columns:
        # don't need rmse for Date
        if col == 'Datetime':
            continue

        key = col
        value = np.sqrt(np.mean((ground_truth_df[col].to_numpy() - df_to_test[col].to_numpy())**2))
        rmse_dict[key] = value #populate dict with rmse

    return rmse_dict

def sMAPE(ground_truth_df, df_to_test, window_start:datetime=None, window_end:datetime=None ):
    ''' 
    1. Checks if all fields and times match up, Throws error if they dont
    2. Calculates sMAPE over time for all fields except "Datetime"
    3. Returns dict with fields as keys and RMSE as values

    '''
    # check if fields match
    if not ground_truth_df.columns.equals(df_to_test.columns):
        raise ValueError("DataFRames have different columns")
    
    # check that dates line up
    if not ground_truth_df["Datetime"].equals(df_to_test["Datetime"]):
        raise ValueError("Dates do not match")
    
    # Apply window if given
    if window_start is not None and window_end is not None:
        mask = (ground_truth_df["Datetime"] >= window_start) & (ground_truth_df["Datetime"] <= window_end)
        ground_truth_df = ground_truth_df[mask].copy()
        df_to_test = df_to_test[mask].copy()
    
    # calculate sMAPE
    sMAPE_dict = {}
    for col in ground_truth_df.columns:
        # don't need rmse for Date
        if col == 'Datetime':
            continue

        key = col
        gt = ground_truth_df[col].to_numpy()
        pre = df_to_test[col].to_numpy()

        value = 100 * np.mean(2 * np.abs(pre -(gt)) / (np.abs(pre) + np.abs(gt)))
        sMAPE_dict[key] = value #populate dict with sMAPE

    return sMAPE_dict


#################################################### functions to mess up original data so we can test our tool ##############################################
def inject_missing_values(df, fields_to_inject_on:list, start_time:datetime, length:timedelta):
    end_time = start_time + length
    df = df.copy()
    mask = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)

    for field in fields_to_inject_on:
        df.loc[mask, field] = np.nan

    return df

def inject_zeros(df, fields_to_inject_on:list, start_time:datetime, length:timedelta):
    end_time = start_time + length
    df = df.copy()
    mask = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)

    for field in fields_to_inject_on:
        df.loc[mask, field] = 0

def inject_noise(df, fields_to_inject_on: list, start_time: datetime, length: timedelta, noise_scale: float = 0.1):
    '''
    noise_scale is the % of the original value that the noise can change the original value
    '''
    end_time = start_time + length
    df = df.copy()
    mask = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)

    for field in fields_to_inject_on:
        original_values = df.loc[mask, field]
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=original_values.shape)
        df.loc[mask, field] = original_values * (1 + noise)

    return df
