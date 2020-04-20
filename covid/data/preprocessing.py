import pandas as pd
import numpy as np
import csv
from dataReader import *
from dataPreProcessing import *
from datetimeCleaner import datetimeCleanerPipe


class DataProcessor:
    """
    Class for reading, processing, and writing data from the
    Semantics scholar website.



    """
    def __init__(self,filename,cols_to_preproc):
        self.filename=filename
        self.df = None
        self.cols_to_preproc = cols_to_preproc
        self.df_preproc = None

    def read_json_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        filenames = getListOfFiles(raw_data_path,extention='json')
        self.df = getCSVPapers(filenames,raw_data_path)

    def read_csv_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        self.df = pd.read_csv(raw_data_path+self.filename)

    def process_data(self):
        """Process raw data into useful files for model."""
        self.df_preproc = nltkPreProcDf(self.df,self.cols_to_preproc)

    def process_dates(self):
        """Normalize dates"""
        self.df_preproc = datetimeCleanerPipe(self.df, 
                                              col='publish_time', 
                                              normalize_future=True)

    def write_data(self, df, processed_data_path):
        """Write processed data to directory."""
        
        df.to_csv(processed_data_path+self.filename,
                    index=False)

