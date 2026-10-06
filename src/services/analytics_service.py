import pandas as pd


class AnalyticsService:
    """Perform analytical operations on a dataset."""

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Dataset cannot be empty.")

        self.df = df

    def count_rows(self) -> int:
        return len(self.df)

    def count_unique(self, column: str) -> int:
        self._validate_column(column)

        return int(self.df[column].nunique())

    def value_counts(self, column: str) -> dict:
        self._validate_column(column)

        return (
            self.df[column]
            .value_counts(dropna=False)
            .to_dict()
        )

    def mean(self, column: str) -> float:
        self._validate_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(
                f"Column '{column}' is not numeric."
            )

        return float(self.df[column].mean())

    def median(self, column: str) -> float:
        self._validate_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(
                f"Column '{column}' is not numeric."
            )

        return float(self.df[column].median())

    def min(self, column: str) -> float:
        self._validate_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(
                f"Column '{column}' is not numeric."
            )

        return float(self.df[column].min())

    def max(self, column: str) -> float:
        self._validate_column(column)

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(
                f"Column '{column}' is not numeric."
            )

        return float(self.df[column].max())

    def _validate_column(self, column: str) -> None:
        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )
    def group_mean(
        self,
        group_column: str,
        value_column: str
    ) -> dict:

        self._validate_column(group_column)
        self._validate_column(value_column)

        if not pd.api.types.is_numeric_dtype(
            self.df[value_column]
        ):
            raise TypeError(
                f"Column '{value_column}' must be numeric."
            )

        result = (
            self.df
            .groupby(group_column)[value_column]
            .mean()
            .sort_values(ascending=False)
        )

        return result.to_dict()

    def group_count(
        self,
        group_column: str
    ) -> dict:

        self._validate_column(group_column)

        result = (
            self.df[group_column]
            .value_counts()
            .to_dict()
        )

        return result
    
    def filter_equals(
        self,
        column: str,
        value
    ) -> pd.DataFrame:

        self._validate_column(column)

        return self.df[
            self.df[column] == value
        ].copy()