# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('filename', type=click.Path())
@click.argument('dataset_name', type=click.Path())
def main(input_filepath, output_filepath, filename, dataset_name):
    """ Merges all datasets into a clean csv file
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')    
    
    df = pd.read_csv(input_filepath+filename,parse_dates=True)

    df_dataset = pd.read_csv(input_filepath+dataset_name,parse_dates=True)

    logger.info('joining data')
    

    df = df.merge(df_dataset,how='left',
                              left_on=['sha','title'],
                              right_on=['paper_id','title'],
                              validate='m:m') 

    logger.info('saving data')
    
    df.to_csv(output_filepath,index=False)

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
