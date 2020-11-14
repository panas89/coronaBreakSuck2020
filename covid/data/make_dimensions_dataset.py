# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from preprocessing import DataProcessor
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('filename', type=click.Path())
@click.argument('only_dates', type=click.Path())
@click.argument('sheet_name', type=click.Path())
@click.argument('colsa', type=click.Path())
@click.argument('colsb', type=click.Path())
def main(input_filepath, output_filepath, filename, only_dates, sheet_name, colsa, colsb):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    cols_to_preproc = None
    preprocessor = DataProcessor(filename=filename, cols_to_preproc=cols_to_preproc)

    logger.info('reading data')
    if 'json' in input_filepath or 'json' in filename:
        preprocessor.read_json_data(input_filepath)
    elif 'xlsx' in input_filepath or 'xlsx' in filename:
        preprocessor.read_xlsx_data(input_filepath,sheet_name)
    else:
        preprocessor.read_csv_data(input_filepath)

    logger.info('processing data')
    if only_dates == 'True':
        preprocessor.process_dates()
    
    ### rename df columns to match original dataset columns names
    colsA = colsa.replace('[','').replace(']','').split(',')
    colsB = colsb.replace('[','').replace(']','').split(',')
    preprocessor.rename_colunms(colsA, colsB)

    logger.info('saving raw data in csv format')
    preprocessor.write_data(preprocessor.df,
                            processed_data_path=output_filepath,without_filename=True)

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
