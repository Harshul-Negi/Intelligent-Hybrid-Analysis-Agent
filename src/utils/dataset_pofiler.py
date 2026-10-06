import pandas as pd


class DatasetProfiler:
    """Generate metadata and statistics for a pandas DataFrame."""

    def profile(self, df: pd.DataFrame) -> dict:
        if df.empty:
            raise ValueError("Cannot profile an empty dataset.")

        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": self._get_dtypes(df),
            "missing_values": self._get_missing_values(df),
            "numeric_columns": self._get_numeric_columns(df),
            "categorical_columns": self._get_categorical_columns(df),
            "unique_values": self._get_unique_values(df),
            "statistics": self._get_statistics(df),
        }

        return profile

    def _get_dtypes(self, df: pd.DataFrame) -> dict:
        return {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        }

    def _get_missing_values(self, df: pd.DataFrame) -> dict:
        return {
            column: int(count)
            for column, count in df.isnull().sum().items()
            if count > 0
        }

    def _get_numeric_columns(self, df: pd.DataFrame) -> list:
        return df.select_dtypes(
            include="number"
        ).columns.tolist()

    def _get_categorical_columns(self, df: pd.DataFrame) -> list:
        return df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def _get_unique_values(self, df: pd.DataFrame) -> dict:
        return {
            column: int(df[column].nunique())
            for column in df.columns
        }

    def _get_statistics(self, df: pd.DataFrame) -> dict:
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return {}

        return numeric_df.describe().to_dict()

    def create_summary(self, profile: dict) -> str:
        summary = []

        summary.append(
            f"Dataset contains {profile['rows']} rows "
            f"and {profile['columns']} columns."
        )

        summary.append(
            "\nColumns:"
        )

        for column in profile["column_names"]:
            dtype = profile["dtypes"][column]
            unique = profile["unique_values"][column]

            summary.append(
                f"- {column}: type={dtype}, unique_values={unique}"
            )

        if profile["missing_values"]:
            summary.append("\nMissing values:")

            for column, count in profile["missing_values"].items():
                summary.append(
                    f"- {column}: {count}"
                )

        return "\n".join(summary)

    def get_schema(self, df: pd.DataFrame) -> list:
        schema = []

        for column in df.columns:
            series = df[column]

            column_info = {
                "name": column,
                "dtype": str(series.dtype),
                "missing": int(series.isnull().sum()),
                "unique_values": int(series.nunique()),
                "sample_values": series.dropna().head(5).tolist(),
            }

            if pd.api.types.is_numeric_dtype(series):
                column_info["semantic_type"] = "numeric"

            elif pd.api.types.is_datetime64_any_dtype(series):
                column_info["semantic_type"] = "datetime"

            else:
                column_info["semantic_type"] = "categorical"

            schema.append(column_info)

        return schema

    def create_schema_summary(self, schema: list) -> str:
        summary = []

        for column in schema:
            summary.append(
                f"- {column['name']} | "
                f"type={column['semantic_type']} | "
                f"dtype={column['dtype']} | "
                f"missing={column['missing']} | "
                f"unique={column['unique_values']} | "
                f"samples={column['sample_values']}"
            )

        return "\n".join(summary)