import os
import re
import pickle
import argparse
import pandas as pd
import numpy as np
from utils import decision, json_pretty_dump
from collections import OrderedDict, defaultdict


seed = 42
np.random.seed(seed)

parser = argparse.ArgumentParser()
# TODO: Not accurate
parser.add_argument("--train_anomaly_ratio", default=0.2, type=float)

params = vars(parser.parse_args())

data_name = f'intrepid_ras_{params["train_anomaly_ratio"]}_tar'
data_dir = "../data/processed/Intrepid_RAS"

params = {
    # "log_file": "../data/HDFS/HDFS.log_structured.csv",
    "log_file": "/Users/artjiang/Documents/2470/Parser-N-Analyzer-for-Intrepid-RAS-log-Dataset/Intrepid_RAS_0901_0908_structured.csv",
    "label_file": "/Users/artjiang/Documents/2470/Parser-N-Analyzer-for-Intrepid-RAS-log-Dataset/Intrepid_RAS_0901_0908_binary_labels.csv",
    "test_ratio": 0.2,
    "random_sessions": True,  # shuffle sessions
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, data_name)
os.makedirs(data_dir, exist_ok=True)


def preprocess_hdfs(
    log_file,
    label_file,
    test_ratio=None,
    train_anomaly_ratio=1,
    random_sessions=False,
    **kwargs
):
    """Load HDFS structured log into train and test data

    Arguments
    ---------
        TODO

    Returns
    -------
        TODO
    """
    print("Loading HDFS logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=True, memory_map=True)
    # struct_log.sort_values(by=["Date", "Time"], inplace=True)
    struct_log.fillna('N/A', inplace=True)
    # assign labels
    label_data = pd.read_csv(label_file, engine="c", na_filter=True, memory_map=True)
    label_data.fillna('N/A', inplace=True)
    label_data["Label"] = label_data["label"].map(lambda x: int(x == "['Abnormal']"))
    label_data_dict = dict(zip(label_data["BLOCK_TIME_UNIQUE_ID"], label_data["Label"]))

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for _, row in enumerate(struct_log.values):
        # blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])
        blkId_list = row[column_idx['BLOCK_TIME_UNIQUE_ID']]
        blkId_set = set([blkId_list])
        for blk_Id in blkId_set:
            if blk_Id not in session_dict:
                session_dict[blk_Id] = defaultdict(list)
            session_dict[blk_Id]["templates"].append(row[column_idx["EventTemplate"]])

    # TODO: Corner Case
    label_data_dict['N/A'] = 0
    for k in session_dict.keys():
        session_dict[k]["label"] = label_data_dict[k]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    session_labels = np.array(list(map(lambda x: label_data_dict[x], session_ids)))

    train_lines = int((1 - test_ratio) * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]
    session_labels_train = session_labels[session_idx_train]
    session_labels_test = session_labels[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["label"] == 0)
        or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))
    }

    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["label"] for k, v in session_train.items()]
    session_labels_test = [v["label"] for k, v in session_test.items()]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))

    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))

    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    preprocess_hdfs(**params)
