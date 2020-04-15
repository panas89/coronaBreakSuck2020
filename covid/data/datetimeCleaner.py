import ast 
from datetime import datetime
import re
import pandas as pd
import numpy as np 

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

    df[col] = df[col].apply(lambda x: ast.literal_eval(x)[0] 
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
    for index, date in raw_dates.items():
        if isinstance(date,list) and date!=[]:
            year, season = tuple(date[0].split())
            raw_date = year + " " + season_to_month[season]
            df.loc[index, col] = raw_date

    return df
            

def convert_to_timestamps(df, col):

    """ Method to convert various differnt date formats to timestamps.
        Replaces input datetime col with Timestamps.
    """

  
    # loop through all pre-defined datetime string formats, 
    # convert values to timestamps and store them in an (df index, timestamp) dict
    clean_dates = {}
    for pattern, DateFormat in pattern_to_DateFormat.items():
        regex = re.compile(pattern, flags=re.IGNORECASE)
        raw_dates = df[col].str.findall(regex).to_dict()
        for index, v in raw_dates.items():
            if isinstance(v,list) and v!=[]: #ignore NaNs (floats)
                raw_date = v[0]
                try:
                    clean_dates[index] = pd.to_datetime(raw_date, format = DateFormat)
                except:
                    pass
    
    # replace raw datetimes of input col with Timestamps
    df[col] = pd.Series(clean_dates)

    return df
    


def datetimeCleanerPipe(df, col):
    """ Method to clean datetime col and convert differnt date formats to timestamps.
        Replaces input datetime col with Timestamps.
    """

    df = convert_list_to_date(df, col)
    df = convert_season_to_month(df, col)
    df = convert_to_timestamps(df, col)

    return df
