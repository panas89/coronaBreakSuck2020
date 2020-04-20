import ast 
from datetime import datetime
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

#----------------------------------
#              INTRO
#----------------------------------

# The different datetime formats of the publish_time col in metadata:
# 0. Y-m-d (e.g. 2020-02-05)
# 1. Y b d (e.g. 2019 Feb 30)
# 2. Y b1 d b2-b3 (e.g. 2011 May 10 Jul-Sep)
# 3. Y b d season (e.g. 2017 Sep 15 Summer)
# 4. Y b1-b2 (e.g. 2006 Jun-Dec)
# 5. Y b (e.g. 2006 May)
# 6. Y season (e.g. 2011 Spring)
# 7. "['2020-02-05', '2020-02']"
# 8. "['2019-09-11', '2020']"
# 9. NaN


#----------------------------------
#             CONSTANTS
#----------------------------------

# Define some handy constants
months_pattern = "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
seasons_pattern = "(?:Winter|Spring|Summer|Fall)"


# Define regex for the different datetime patterns
Y_m_d = '\d{4}[/-]\d{1,2}[/-]\d{1,2}'
Y_b_d = '\d{4}[ ]' + months_pattern + '[ ]\d{1,2}'
Y_b_d_b_b = '(\d{4}[ ]' + months_pattern + '[ ]\d{1,2})[ ]' +  months_pattern + '[\-]' + months_pattern
Y_b_d_s = '(\d{4}[ ]' + months_pattern + '[ ]\d{1,2})[ ]' + seasons_pattern
Y_b_b = '(\d{4}[ ]' + months_pattern + ')[/-]' + months_pattern
Y_b = '\d{4}[ ]' + months_pattern +'$'
Y_s = '\d{4}[ ]' + seasons_pattern 

pattern_to_DateFormat = {Y_m_d: '%Y-%m-%d',
                      Y_b_d: '%Y %b %d',
                      Y_b_d_b_b: '%Y %b %d',
                      Y_b_d_s: '%Y %b %d',
                      Y_b_b: '%Y %b',
                      Y_b: '%Y %b'
                     }


#----------------------------------
#             FUNCTIONS
#----------------------------------

def convert_list_to_date(df, col):
    """ Method to convert string representation of list of dates to a single date.
    e.g. "['2020-02-05', '2020-02']" -----> '2020-02-05'
    """

    df[col] = df[col].progress_apply(lambda x: ast.literal_eval(x)[0] 
                                      if isinstance(x,str) and x.startswith("[") 
                                      else x)
    return df

def convert_season_to_month(df, col):
    """ Method to convert season in date to month date. 
        We substitute season with mid-season month.
    e.g. '2011 Spring' -----> '2011 Apr'
    """

    season_to_month = {'Winter': 'Jan', 
                       'Spring': 'Apr', 
                       'Summer': 'Jul', 
                       'Fall': 'Oct'}

    # find dates with format: Year Season 
    regex = re.compile(Y_s, flags=re.IGNORECASE)
    raw_dates = df[col].str.findall(regex).to_dict()

    # replace Season with month value from season_to_month dict
    for index, date in tqdm(raw_dates.items()):
        if isinstance(date,list) and date!=[]:
            year, season = tuple(date[0].split())
            raw_date = year + " " + season_to_month[season]
            df.loc[index, col] = raw_date

    return df
            

def convert_to_timestamps(df, col):

    """ Method to convert various differnt date formats to timestamps.
        Replaces input datetime col with Timestamps.
    """
    NUM_PAPERS = len(df)
    NUM_NULLS = df.publish_time.isnull().sum()
    NUM_BAD_TIMESTAMPS = pd.to_datetime(df.publish_time).isnull().sum() - NUM_NULLS

    print("Fraction of papers with MISSING dates: {}/{}".format(NUM_NULLS, NUM_PAPERS))
    print("Fraction of papers with BAD dates BEFORE pre-processing: {}/{}"
          .format(NUM_BAD_TIMESTAMPS, NUM_PAPERS))

    # loop through all pre-defined datetime string formats, 
    # convert values to timestamps and store them in an (df index, timestamp) dict
    clean_dates = {}
    for pattern, DateFormat in tqdm(pattern_to_DateFormat.items()):
        regex = re.compile(pattern, flags=re.IGNORECASE)
        raw_dates = df[col].str.findall(regex).to_dict()
        for index, v in raw_dates.items():
            if isinstance(v,list) and v!=[]: #ignore NaNs (floats)
                raw_date = v[0]
                try:
                    clean_dates[index] = pd.to_datetime(raw_date, format = DateFormat, errors='coerce')
                except:
                    pass
    
    # replace raw datetimes of input col with Timestamps
    df[col] = pd.to_datetime(pd.Series(clean_dates))

    NUM_BAD_TIMESTAMPS = df.publish_time.isnull().sum() - NUM_NULLS
    print("Fraction of papers with BAD dates AFTER pre-processing: {}/{}"
          .format(NUM_BAD_TIMESTAMPS, NUM_PAPERS))

    return df
    
def normalize_future_dates(df, col):
    """ Method to set the date of all papers with timestamps in the future to today.
    e.g. 2020-12-31 ----> today
    """

    # collect ids of papers with timestamps in the future
    today = pd.Timestamp(datetime.date(datetime.now()))
    future_date_ids = df[df[col] > today].index

    print("Fraction of papers with FUTURE dates: {}/{}".format(len(future_date_ids), len(df)))

    # replace future dates with today
    df.loc[future_date_ids, col] = today

    return df


def datetimeCleanerPipe(df, col, normalize_future=True):
    """ Method to clean datetime col and convert differnt date formats to timestamps.
        Replaces input datetime col with Timestamps.
    """

    try:
        df = convert_list_to_date(df, col)
    except:
        raise Exception("Something went bad with: convert_list_to_date.")

    try:
        df = convert_season_to_month(df, col)
    except:
        raise Exception("Something went bad with: convert_season_to_month.")

    try:
        df = convert_to_timestamps(df, col)
    except:
        raise Exception("Something went bad with: convert_to_timestamps.")
  

    # set value of future timestamps to today
    if normalize_future:
        try:
            df = normalize_future_dates(df, col)
        except:
            raise Exception("Something went bad with: normalize_future_dates.")
    

    return df
