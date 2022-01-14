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

    def __init__(self, filename, cols_to_preproc):
        self.filename = filename
        self.df = None
        self.cols_to_preproc = cols_to_preproc
        self.df_preproc = None

    def read_json_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        filenames = getListOfFiles(raw_data_path, extention="json")
        self.df = getCSVPapers(filenames, raw_data_path)

    def read_csv_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        self.df = pd.read_csv(raw_data_path + self.filename, engine="python")

    def read_xlsx_data(self, raw_data_path, sheet_name):
        """Read raw data into DataProcessor."""
        self.df = pd.read_excel(raw_data_path + self.filename, sheet_name=sheet_name)

    def rename_colunms(self, colsA, colsB):
        """Rename dataframe columns."""
        self.df.rename(
            {colA: colB for colA, colB in zip(colsA, colsB)}, axis=1, inplace=True
        )

    def process_data(self):
        """Process raw data into useful files for model."""
        self.df_preproc = nltkPreProcDf(self.df, self.cols_to_preproc)

    def process_dates(self, col="publish_time"):
        """Normalize dates"""
        self.df_preproc = datetimeCleanerPipe(self.df, col=col, normalize_future=True)

    def write_data(self, df, processed_data_path, without_filename=False):
        """Write processed data to directory."""
        if without_filename:
            df.to_csv(processed_data_path, index=False)
        else:
            df.to_csv(processed_data_path + self.filename, index=False)
