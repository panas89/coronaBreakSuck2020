# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from preprocessing import DataProcessor


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('filename', type=click.Path())
@click.argument('cols_to_preproc', type=click.Path())
@click.argument('only_dates', type=click.Path())
def main(input_filepath, output_filepath, filename, cols_to_preproc, only_dates):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    cols_to_preproc = cols_to_preproc.replace('[','').replace(']','').split(',')
    preprocessor = DataProcessor(filename=filename,
                                 cols_to_preproc=cols_to_preproc)

    logger.info('reading data')
    if 'json' in input_filepath:
        preprocessor.read_json_data(input_filepath)
    else:
        preprocessor.read_csv_data(input_filepath)


    logger.info('processing data')
    if only_dates == 'True':
        preprocessor.process_dates()
    else:
        preprocessor.process_data()

    logger.info('saving raw data in csv format')
    preprocessor.write_data(preprocessor.df,
                            processed_data_path=output_filepath.replace('processed','raw'))

    logger.info('saving processed data')
    preprocessor.write_data(preprocessor.df_preproc,
                            processed_data_path=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
