# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from covid.data import utils as data_utils
from tqdm import tqdm
from joblib import Parallel, delayed
tqdm.pandas()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('num_cores', type=click.Path())
def main(input_filepath, output_filepath,num_cores):
    """ Runs data processing scripts to turn raw merged data from (../raw) into
        cleaned covid papers ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('reading data')

    df = pd.read_csv(input_filepath)

    logger.info('Pre-processing location ...')
    # 
    cols = ['affiliations', 'location']

    def classify_countries(text):
        # print(text)
        if pd.isnull(text):
            return np.nan
        else:
            # classification
            countries = data_utils.extract_location(text, is_robust=True)
            return countries

    
    num_cores = int(num_cores)

    # loop through each column's address and make classification
    for col in cols:
        print(col)
        preds = Parallel(n_jobs=num_cores)(delayed(classify_countries)(i) for i in tqdm(df[col].tolist()))

        # assign
        df['%s_country'%col] = preds

    # Save data
    df.to_csv(output_filepath, index=False)
        


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
