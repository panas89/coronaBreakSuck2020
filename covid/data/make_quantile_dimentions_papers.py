# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from preprocessing import DataProcessor
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("filename", type=click.Path())
@click.argument("classified_filepath", type=click.Path())
@click.argument("sheet_name", type=click.Path())
@click.argument("lb", type=click.Path())
@click.argument("ub", type=click.Path())
def main(
    input_filepath, output_filepath, filename, classified_filepath, sheet_name, lb, ub
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading raw data ...")

    if "xlsx" in input_filepath or "xlsx" in filename:
        df = pd.read_excel(input_filepath + filename, sheet_name=sheet_name)
    else:
        df = pd.read_csv(input_filepath + filename, engine="python")

    logger.info("Reading classified data ...")
    df_classified = pd.read_csv(classified_filepath)

    logger.info("Getting papers in high impact range of " + lb + " - " + ub + " ... ")
    lb, ub = float(lb), float(ub)
    low, high = df["Altmetric"].quantile([lb, ub])
    df_quantile = df.query(
        "{low}<Altmetric<{high}".format(low=low, high=high)
    ).reset_index(drop=True)

    logger.info("Keeping only high impact papers ...")
    cols = ["Publication ID", "Publisher", "Source title", "Altmetric"]
    df_hi = df_classified.merge(
        df_quantile.loc[:, cols], how="inner", left_on="sha", right_on="Publication ID"
    )

    logger.info("Saving data in csv format ...")
    df_hi.to_csv(output_filepath, index=False)

    return


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
