"""
This python script is prepared to automatically run the relation extraction for the classes 
and subclasses. The resulting file will be a .csv file

This script follows notebooks/main_kch_relation_extraction.ipynb
"""
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from covid.models.relation.extraction import RelationExtractor

# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('yaml_filepath', type=click.Path())
def main(input_filepath, output_filepath, yaml_filepath):
    """ 
        Runs relationship extraction script and identifies relations ships relation to covid
    """

    logger = logging.getLogger(__name__)
    logger.info('Reading preprocessed data ...')

    # Note
    # for how the data files look like, 
    # checkout here: https://drive.google.com/drive/u/1/folders/1LSNRnk24Uiqb-l2Sm_0Uz-Fa2hI5YzYi

    # initiate the extractor
    print('----- Initiating the extractor -----')
    covidre = RelationExtractor(km_path=yaml_filepath)#'../covid/models/paperclassifier/interest.yaml')

    # load the paper-classified csv <-- paper that have been identified with classes
    print('----- Loading the paper-classified csv -----')
    df = pd.read_csv(input_filepath)#'../data/paperclassifier/classified_merged_covid.csv')

    # extraction all the information
    # it will take a while, ~10mins
    print('----- Extracting the relation information -----')
    relations = covidre.extract_all(df)
    df['relations'] = relations

    # save the information
    print('----- saving -----')
    df.to_csv(output_filepath)#'../data/paperclassifier/classified_merged_covid_relation.csv')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
