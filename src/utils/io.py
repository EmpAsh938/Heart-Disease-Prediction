from pathlib import Path

import joblib


def ensure_directory(path):

    Path(path).mkdir(

        parents=True,

        exist_ok=True

    )


def save_pickle(obj, path):

    ensure_directory(Path(path).parent)

    joblib.dump(obj, path)


def load_pickle(path):

    return joblib.load(path)