from pathlib import Path
import pandas as pd


class DatasetLoader:
    """
    Loads datasets into pandas DataFrames.
    """

    def load_csv(self, file_path: str) -> pd.DataFrame:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")

        try:
            df = pd.read_csv(path)

        except Exception as e:
            raise Exception(f"Unable to load CSV.\n{e}")

        if df.empty:
            raise ValueError("Dataset is empty.")

        return df