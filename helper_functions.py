import pandas as pd

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

import matplotlib.pyplot as plt

def plot_all_fields(df, title):
    '''
    This is an edited chat gpt function. Gpt 04 Prompt: "s there a good way to plot each field in a df against the time in a long subplot that has one column and lots of rows"
    '''
    time_column = 'Datetime'

    num_fields = df.shape[1] - 1  # Exclude time column
    fields = [col for col in df.columns if col != time_column]

    fig, axes = plt.subplots(nrows=num_fields, ncols=1, figsize=(12, 0.5 * num_fields), sharex=True)

    # If only one axis (just one field), make it iterable
    if num_fields == 1:
        axes = [axes]

    for ax, field in zip(axes, fields):
        ax.plot(df[time_column], df[field])
        ax.set_ylabel(field)
        ax.grid(True)

    axes[-1].set_xlabel(time_column)
    fig.tight_layout()
    plt.title(title)
    plt.show()
