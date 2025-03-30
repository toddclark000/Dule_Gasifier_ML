import pandas as pd
import matplotlib.pyplot as plt

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
    Plots each non-time column in a vertical stack of subplots vs. the 'Datetime' column.
    '''
    time_column = 'Datetime'

    num_fields = df.shape[1] - 1  # Exclude time column
    fields = [col for col in df.columns if col != time_column]

    fig, axes = plt.subplots(
        nrows=num_fields,
        ncols=1,
        figsize=(12, 0.9 * num_fields),  # Slightly taller per row
        sharex=True
    )

    # Ensure axes is always iterable
    if num_fields == 1:
        axes = [axes]

    for ax, field in zip(axes, fields):
        ax.plot(df[time_column], df[field])
        ax.set_title(field, fontsize=10)
        ax.grid(True)

    axes[-1].set_xlabel(time_column)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # Reserve space for suptitle
    fig.subplots_adjust(hspace=0.5)  # Add vertical spacing between subplots
    plt.show()

def get_relevant_fields(df):

    column_filter = ['Date', 'Time', 'EFG Air Flow Rate lb/hr', 'Primary O2 Flow Rate lb/hr',
                    'EFG System Pressure psi',
                    'Slurry Flow Rate lb/hr',
                    'EFG:EFG_Oxygen_Pressure_1.Value',
                    'EFG:EFG_Oxygen_Pressure_2.Value',
                    ###############################################################################
                    'Analyzer 2 CO2 %',                 # Is it worth keeping these extra analyzers?
                    'Analyzer 2 CO %',                  # Tf they are just being kept because they
                    'Analyzer 3 O2 %',                  # have errors than that is exactly what we
                    'Analyzer 3 CO2%',                  # want.
                    'ABB CH4',
                    'ABB CO',
                    'ABB CO2',
                    'ABB O2',
                    ###############################################################################
                    'GB O2%',
                    'GB CO2%',
                    'GB CO%',
                    'GB CH4%',
                    'GB H2%',
                    'GB C2H4%',
                    'GB C2H2%',
                    'GB LHV [MJ/Nm3]',
                    'EFG:B_TC_1_Shallow.Value',
                    'EFG:B_TC_4_Shallow.Value'
                    ]
    
    if 'Moyno Flow Rate lb/hr' in df.columns:
        df.rename(columns={'Moyno Flow Rate lb/hr': 'Slurry Flow Rate lb/hr'}, inplace=True)
    
    return df[column_filter]