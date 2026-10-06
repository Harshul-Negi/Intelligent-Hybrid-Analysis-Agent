from langchain_core.tools import tool

from services.analytics_service import AnalyticsService


def create_analytics_tools(
    analytics: AnalyticsService
):
    """Create tools backed by the analytics service."""

    @tool
    def calculate_mean(column: str) -> float:
        """
        Calculate the average of a numeric column.

        Use this when the user asks for the average or mean
        of a numerical column.
        """
        return analytics.mean(column)

    @tool
    def calculate_median(column: str) -> float:
        """
        Calculate the median of a numeric column.

        Use this when the user asks for the median
        of a numerical column.
        """
        return analytics.median(column)

    @tool
    def calculate_min(column: str) -> float:
        """
        Find the minimum value of a numeric column.
        """
        return analytics.min(column)

    @tool
    def calculate_max(column: str) -> float:
        """
        Find the maximum value of a numeric column.
        """
        return analytics.max(column)

    @tool
    def count_unique(column: str) -> int:
        """
        Count the number of unique values in a column.
        """
        return analytics.count_unique(column)

    @tool
    def get_value_counts(column: str) -> dict:
        """
        Count how frequently each value appears in a column.
        """
        return analytics.value_counts(column)

    @tool
    def calculate_group_mean(
        group_column: str,
        value_column: str
    ) -> dict:
        """
        Calculate the average value_column for each
        group in group_column.
        """
        return analytics.group_mean(
            group_column,
            value_column
        )

    @tool
    def calculate_group_count(
        group_column: str
    ) -> dict:
        """
        Count the number of records in each group.
        """
        return analytics.group_count(group_column)

    @tool
    def calculate_correlation(
        column_a: str,
        column_b: str
    ) -> float:
        """
        Calculate Pearson correlation between two
        numeric columns.
        """
        return analytics.correlation(
            column_a,
            column_b
        )

    return [
        calculate_mean,
        calculate_median,
        calculate_min,
        calculate_max,
        count_unique,
        get_value_counts,
        calculate_group_mean,
        calculate_group_count,
        calculate_correlation,
    ]