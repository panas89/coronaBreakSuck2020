# Misc
import os
import pickle
from tqdm import tqdm
from pathlib import Path
import yaml

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Arrays & Dataframes
import numpy as np
import pandas as pd

# Utilitieswhic
from covid.models.topicmodeling.ldamodel import LDAModel
from covid.models.topicmodeling.utilities.constants import (
    COMMON_PHRASES_REGEX,
    COMMON_WORDS,
)
from covid.models.topicmodeling.utilities.functions import *

# Gensim
from gensim.models import Phrases

# -------------------------- Global Vars/Functions ----------------------------

TOP_DIR = str(Path.cwd())
LOWER_TOPIC_BOUND = 2  # no less than 2 topics
UPPER_TOPIC_BOUND = 8  # no more than 12 topics


def range_num_topics(num_papers, lower=None, upper=None):
    if not upper:
        upper = max(2, num_papers // 10) + 1
        upper = min(upper, 12)
    if not lower:
        lower = upper // 2

    return list(range(lower, upper, 1))


def learn_topics(df, class_col, output_filename, train_on_col="clean_text"):

    # 1. Prepare Text Data
    #######################

    print(f"1. Preparing Text Data\n for {class_col}")

    NUM_ALL_PAPERS = len(df)

    # Query class/subclass col and then drop it
    df = df.loc[df[class_col] == 1].drop([class_col], axis=1)

    NUM_PAPERS = len(df)
    print("Fraction of {} papers: {}/{}".format(class_col, NUM_PAPERS, NUM_ALL_PAPERS))

    text_data = df[train_on_col].values.tolist()

    # Add bigrams to text data
    bigram_model = Phrases(
        text_data, min_count=3, threshold=50
    )  # higher threshold fewer phrases.
    text_data = [bigram_model[doc] for doc in text_data]

    # 2. Train LDA Models
    #######################

    ###########setting Lower_topic_bound and upper_topic_bound

    # initialize LDA model
    temp_lda = LDAModel(text_data, test_data=None)

    ####### lower and upper topics bounds  +- 2 space to search

    LOWER_TOPIC_BOUND = temp_lda.getLowerTopicBound(upper_bound=8)
    UPPER_TOPIC_BOUND = (
        LOWER_TOPIC_BOUND + 2
    )  # use small range to speed up parameter tuning

    if LOWER_TOPIC_BOUND > 4:  # to subtract 2
        LOWER_TOPIC_BOUND = LOWER_TOPIC_BOUND - 2

    print("\n\n2. Starting Training\n")
    # Choose training parameters
    param_grid_mallet = {
        "num_topics": range_num_topics(
            NUM_PAPERS, lower=LOWER_TOPIC_BOUND, upper=UPPER_TOPIC_BOUND
        ),
        "iterations": [1000],
        "random_seed": [100],
        "workers": [1],
    }

    tfidf_grid = [0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.90]

    scorers = ["coherence_score"]

    # Train Mallet LDA models
    all_models = []
    for fraction in tqdm(tfidf_grid):
        # initialize LDA model
        temp_lda = LDAModel(text_data, test_data=None).filter_low_tfidf(
            text_data, keep_fraction=fraction
        )
        # run grid-search: automatically assigns best model to LDAModel
        _ = temp_lda.grid_search(
            text_data, param_grid=param_grid_mallet, lda_class="mallet", scorers=scorers
        )

        all_models.append(temp_lda)

    # Print results
    for i, model in enumerate(all_models):
        print(f"\nTF-IDF fraction: {tfidf_grid[i]}")
        print(
            "- Coherence score: {}\n- Number of topics: {}".format(
                model.coherence_score, model.params["num_topics"]
            )
        )

    # Find best LDA model
    all_models.sort(key=lambda model: model.coherence_score)
    best_model = all_models[-1]
    print(
        "\nSelected Model has\n- Coherence score: {}\n- Number of topics: {}".format(
            best_model.coherence_score, best_model.params["num_topics"]
        )
    )

    # 3. Create Topic-Tags
    #######################

    df_topics = best_model.create_top_topic_per_doc_df(best_model.corpus)

    # replace index (= range(len(df_top_topic_per_doc))) with the index values from
    # the paperclassified df
    df_topics.index = df.index

    df = df.join(df_topics)

    print("\n3. Created topic-tags")

    # 4. Save Data
    ################

    SAVE_DIR = TOP_DIR + f"/data/topicmodels/{output_filename}" + f"/{class_col}/"
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    df.to_csv(SAVE_DIR + "data.csv", index=False)

    with open(SAVE_DIR + "model.pickle", "wb") as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(SAVE_DIR + "vis.pickle", "wb") as handle:
        pickle.dump(
            best_model.getLDAVisualization(), handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"\n4. Saved Classified Papers and Best LDA Models with Vis in {SAVE_DIR}")

    return df_topics.loc[:, ["dominant_topic", "topic_keywords"]]


# -------------------------- MAIN -------------------------


@click.command()
@click.argument("yaml_filepath", type=click.Path())
@click.argument("input_filename", type=click.Path())
@click.argument("output_filename", type=click.Path())
@click.argument("start_date", type=click.Path())
def main(yaml_filepath, input_filename, output_filename, start_date):

    print("Loading & Cleaning The Data\n")

    # Load paperclassified data
    file_path = TOP_DIR + "/data/processed/" + input_filename + ".csv"
    df = pd.read_csv(file_path, parse_dates=["publish_time"])

    print("Keeping only entries after " + start_date)
    df = process_pcf_data(
        df,
        bad_phrases=COMMON_PHRASES_REGEX,
        bad_tokens=COMMON_WORDS,
        clean_col="abstract",
        drop_nan_text=True,
        from_date=start_date,
    )

    # Obtain class/subclass strings from yaml
    yaml_path = TOP_DIR + "/" + yaml_filepath
    with open(yaml_path) as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    CLASSES = list(yaml_dict.keys())
    CLASSES.remove("disease_name")  # already used for Covid selection
    SUBCLASSES = []
    for c in CLASSES:
        subclasses_of_c = list(yaml_dict[c].keys())
        SUBCLASSES.extend(subclasses_of_c)

    # creating main folder
    if not os.path.exists(TOP_DIR + f"/data/topicmodels/{output_filename}/"):
        os.mkdir(TOP_DIR + f"/data/topicmodels/{output_filename}/")

    # Learn topics for each class/subclass
    bad_cols = []
    for class_col in CLASSES + SUBCLASSES:
        try:
            # Learn topics for papers with class_col tag
            df_topics = learn_topics(
                df, class_col, output_filename, train_on_col="clean_abstract"
            )
            # Append topic-tags/topic-keywords cols to df
            df = df.join(df_topics)
            df.loc[:, "dominant_topic"] = df["dominant_topic"].fillna(-1).astype("int")
            df.loc[:, "topic_keywords"] = df["topic_keywords"].fillna(0)
            df.rename(
                {
                    "dominant_topic": class_col + "_topic",
                    "topic_keywords": class_col + "_topic_kw",
                },
                axis=1,
                inplace=True,
            )
        except Exception as e:
            print(e)
            # break
            bad_cols.append(class_col)

    print("Bad columns:", bad_cols)

    df.to_csv(
        TOP_DIR
        + f"/data/topicmodels/{output_filename}/pcf_"
        + output_filename
        + "_topic_data.csv",
        index=False,
    )
    print(
        "\n\nPath of Final Classified DF\n"
        + "-" * 27
        + "\n\n"
        + TOP_DIR
        + "/data/topicmodels/"
        + output_filename
        + "/pcf_"
        + output_filename
        + "_topic_data.csv"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
