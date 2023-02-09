# add spark context
from pyspark.sql import SparkSession
import pytest


@pytest.fixture(scope='session')
def spark() -> SparkSession:
    """
    Create a local spark session for unit testing.
    Returns:
        spark: SparkSession
    """
    return (
        SparkSession.builder
        .master("local")
        .appName("pytest-pyspark-local")
        .getOrCreate()
    )
