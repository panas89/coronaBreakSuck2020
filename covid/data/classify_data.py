# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from covid.models.paperclassifier.paperclassifier import PaperClassifier


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("yaml_filepath", type=click.Path())
@click.argument("use_cols", type=click.Path())
def main(input_filepath, output_filepath, yaml_filepath, use_cols):
    """Runs data processing scripts to turn raw merged data from (../raw) into
    cleaned covid papers ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("reading data")
    print(input_filepath)

    USE_COLS = use_cols.replace("[", "").replace("]", "").split(",")
    df = pd.read_csv(input_filepath, usecols=USE_COLS).rename(
        {"abstract_x": "abstract"}, axis=1
    )
    NUM_PAPERS = len(df)

    logger.info("Replacing invalid dates to NaN")

    df["publish_time"] = [
        pub_time
        if pd.notnull(pub_time) and isinstance(pub_time, str) and len(pub_time) > 7
        else np.nan
        for pub_time in df["publish_time"]
    ]

    logger.info("Dropping rows with Null dates --  to speed up computation")
    df.dropna(subset=["publish_time"], inplace=True)

    logger.info("Classifying ...")
    # Load the paperclassifier
    print("Yaml file:", yaml_filepath)
    pc = PaperClassifier(km_path=yaml_filepath)

    # Preprocess the dataframe text
    df_p = pc.preprocess(df)

    # Classify papers: add tags
    df_p = pc.classify_all(df_p)

    # Select covid data
    df_covid = df_p.loc[df_p["covid_related"] == 1].drop(["covid_related"], axis=1)
    NUM_COVID_PAPERS = len(df_covid)

    print("Fraction of covid papers: {}/{}".format(NUM_COVID_PAPERS, NUM_PAPERS))

    # Save data
    df_covid.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
