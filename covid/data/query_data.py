# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from covid.models.query_model import *
from covid.data.constants import *
from covid.models.paperclassifier.frontpaperclassifier import FrontPaperClassifier


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('filename', type=click.Path())
@click.argument('cols_to_query', type=click.Path())
@click.argument('yaml_path', type=click.Path())
@click.argument('subclass', type=click.Path())
def main(input_filepath, output_filepath, filename,cols_to_query,yaml_path,subclass):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('generating risk factor and covid patterns')
    cols_to_query = cols_to_query.replace('[','').replace(']','').split(',')

    print(yaml_path)
    # Instantiate FrontPaperClassifier; to be used for keywords retrieval
    fpc = FrontPaperClassifier(km_path=yaml_path)

    #####################################################
    # Define keyword patterns to filter papers out
    cp = PatternGenerator(words=COVID_WORDS)
    cp.generatePattern()

    rp = PatternGenerator(words=fpc.get_keywords(subclass))
    rp.generatePattern()
    
    patterns = [rp.getPattern(),cp.getPattern()]

    
    logger.info('quering dataset' + filename)

    #####################################################
    ## query data
    queror = DataSearchByQueryEngine(filename=filename,
                                     cols_to_query=cols_to_query,
                                     patterns=patterns)

    queror.read_data(input_filepath+filename)

    queror.query_data()


    logger.info('saving filtered data')
    queror.write_data(queror.df_filtered,
                      processed_data_path=output_filepath)
    logger.info('Final number of filtered papers: ' +str(len(queror.df_filtered)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
