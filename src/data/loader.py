from pathlib import Path

import pandas as pd


class DataLoader:

    def __init__(self, path: Path):

        self.path = Path(path)

        self._data = None

    @property
    def data(self):

        if self._data is None:

            self.load()

        return self._data

    def load(self):

        self._data = pd.read_csv(self.path)

        return self._data

    def shape(self):

        return self.data.shape

    def columns(self):

        return self.data.columns.tolist()

    def head(self, n=5):

        return self.data.head(n)

    def describe(self):

        return self.data.describe()

    def info(self):

        return self.data.info()

    def copy(self):

        return self.data.copy()