import pandas as pd
import numpy as np
import glob
from tqdm import tqdm


if __name__ == "__main__":
    path = "data/dashDatasets/"
    folders = [
        "nre/",
        "topicmodeling/",
    ]
    for folder in folders:
        print("Folder: ", folder)
        filenames = [
            filename.replace(path + folder, "")
            for filename in glob.glob(path + folder + "/*.csv")
        ]

        for filename in tqdm(filenames):
            print("Filename: ", filename)

            df = pd.read_csv(path + folder + filename)

            locations = []
            if "semantic" in filename and folder == "topicmodeling/":
                location_series = df["location_country"]
            else:
                location_series = df["location"]

            for location in location_series:
                if pd.isnull(location):
                    locations.append(location)
                elif ";" in location:
                    locations.append(location.split(";")[0])
                elif "," in location:
                    locations.append(location.split(",")[0])
                else:
                    locations.append(location)

            df["location"] = locations

            df.to_csv(path + folder + filename)