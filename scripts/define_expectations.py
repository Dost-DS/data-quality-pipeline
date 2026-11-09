"""Define the expectation suite for the TLC taxi schema. 

Covers three groups of rule: schema presence, non-null constraints on the
fields the pipeline keys on, and value constraints on ranges and category
sets. Tolerances (`mostly`) allow a small proportion of genuine outliers
without failing an entire batch.
"""
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
)

SUITE_NAME = "taxi_trip_suite"

COLUMNS = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag",
    "PULocationID", "DOLocationID", "payment_type", "fare_amount", "extra",
    "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge",
    "total_amount", "congestion_surcharge", "Airport_fee",
]

context = gx.get_context(mode="file", project_root_dir="great_expectations")
suite = ExpectationSuite(name=SUITE_NAME)

# Schema presence
for col in COLUMNS:
    suite.add_expectation(ExpectColumnToExist(column=col))

# Non-null on the fields the pipeline keys on
for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime",
            "PULocationID", "DOLocationID"]:
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

# Timestamps must fall inside the study year
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="tpep_pickup_datetime",
    min_value="2024-01-01T00:00:00", max_value="2025-01-01T00:00:00",
    mostly=0.999))
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="tpep_dropoff_datetime",
    min_value="2024-01-01T00:00:00", max_value="2025-01-02T00:00:00",
    mostly=0.999))

# Numerical ranges. Negative fares and totals occur as refunds and
# adjustments in this dataset, so the lower bounds are not zero.
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="passenger_count", min_value=0, max_value=8, mostly=0.99))
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="trip_distance", min_value=0, max_value=200, mostly=0.98))
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="fare_amount", min_value=-200, max_value=1000, mostly=0.98))
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="total_amount", min_value=-200, max_value=1500, mostly=0.98))

# Category sets, per the TLC data dictionary
suite.add_expectation(ExpectColumnValuesToBeInSet(
    column="VendorID", value_set=[1, 2, 6, 7], mostly=0.99))
suite.add_expectation(ExpectColumnValuesToBeInSet(
    column="payment_type", value_set=[0, 1, 2, 3, 4, 5, 6], mostly=0.99))
suite.add_expectation(ExpectColumnValuesToBeInSet(
    column="store_and_fwd_flag", value_set=["Y", "N"], mostly=0.95))

context.suites.add(suite)
print(f"suite '{SUITE_NAME}': {len(suite.expectations)} expectations")
