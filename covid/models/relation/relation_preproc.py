import vis
import pandas as pd
import numpy as np
import pickle
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from covid.models.paperclassifier.meshsearch import *
import glob


if __name__ == "__main__":
    path = "data/paperclassifier/"
    filenames = [
        filename.replace(path, "")
        for filename in glob.glob(path + "/*relation.csv")
        if "pre_proc" not in filename
    ]

    for filename in filenames:

        df = pd.read_csv(path + filename)

        mesh_obj_path = "covid/models/paperclassifier/mesh_obj.pkl"

        with open(mesh_obj_path, "rb") as f:
            mesh_obj = pickle.load(f)

        kw_lists = list(mesh_obj.id2keywords.values())

        df_new = vis.preprocess_df(df)

        new_kws = []

        for kw in df_new["keyword"]:
            kw_list_to_keep = []
            for kw_list in kw_lists:
                if kw in kw_list:
                    kw_list_to_keep = kw_list

            if len(kw_list_to_keep) > 0:
                new_kws.append(kw_list_to_keep[0])
            else:
                new_kws.append(np.nan)

        df_new["keyword"] = new_kws

        df_new.drop_duplicates(inplace=True)

        df_new.to_csv(path + "pre_proc_" + filename)
