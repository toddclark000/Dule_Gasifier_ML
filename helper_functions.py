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
    
    return df

