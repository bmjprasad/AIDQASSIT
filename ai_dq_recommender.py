 """
 AI‑Assisted Data Quality (DQ) Rule Recommender for Delta Tables

 This PySpark script profiles a Delta table, recommends data quality rules
 based on metadata and sample profiling, optionally refines them with simple
 ML/historical signals, and writes the results to a Delta DQ table.

 Quick start (Databricks):
   - Attach to a cluster with Delta enabled (Databricks default).
   - Run with %run or as a job, e.g.:
       %sh
       python /Workspace/Repos/.../ai_dq_recommender.py \
         --input-type view --input-value demo_orders_view \
         --dq-rules-path dbfs:/tmp/delta/dq_rules --demo

 Quick start (OSS Spark):
   - Provide Delta Lake JARs on the Spark classpath.
   - spark-submit with Delta extensions, e.g.:
       spark-submit \
         --packages io.delta:delta-spark_2.12:3.2.0 \
         --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
         --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
         ai_dq_recommender.py \
         --input-type path --input-value /tmp/delta/demo_orders \
         --dq-rules-path /tmp/delta/dq_rules --demo

 What it does:
   1) Loads table and collects metadata (schema, nullable, comments).
   2) Profiles columns from a sample: nulls, distincts, min/max, percentiles,
      lengths, and simple regex pattern fractions.
   3) Recommends rules: NOT NULL, UNIQUE, ENUM, RANGE, PATTERN.
   4) Optionally refines confidence using simple historical profiles and
      allows an optional LLM hook to incorporate business descriptions.
   5) Writes recommended rules to a Delta DQ table.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# --------------- Spark / Delta setup ---------------

def get_spark(app_name: str = "AI-DQ-Recommender") -> SparkSession:
    """Create or return a SparkSession with Delta support if not on Databricks."""
    is_databricks = os.environ.get("DATABRICKS_RUNTIME_VERSION") is not None
    if is_databricks:
        # Databricks provides a configured SparkSession named `spark`.
        try:
            # type: ignore[name-defined]
            return spark  # noqa: F821
        except NameError:  # running as script on DBR
            return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )


# --------------- Configuration model ---------------

class DQConfig:
    def __init__(
        self,
        input_type: str,
        input_value: str,
        dq_rules_path: str,
        sample_fraction: float = 0.25,
        sample_limit: int = 200_000,
        enum_max_distinct: int = 20,
        enum_max_ratio: float = 0.05,
        unique_tolerance: float = 0.01,
        percentile_range: Tuple[float, float] = (0.01, 0.99),
        profiles_history_path: str | None = None,
        enable_llm: bool = False,
    ) -> None:
        self.input_type = input_type  # one of: table | path | view
        self.input_value = input_value
        self.dq_rules_path = dq_rules_path
        self.sample_fraction = sample_fraction
        self.sample_limit = sample_limit
        self.enum_max_distinct = enum_max_distinct
        self.enum_max_ratio = enum_max_ratio
        self.unique_tolerance = unique_tolerance
        self.percentile_range = percentile_range
        self.profiles_history_path = profiles_history_path
        self.enable_llm = enable_llm


# --------------- Load + metadata utilities ---------------

def load_input_df(spark: SparkSession, input_type: str, input_value: str) -> Tuple[str, DataFrame]:
    if input_type == "table":
        df = spark.table(input_value)
        identity = input_value
    elif input_type == "path":
        df = spark.read.format("delta").load(input_value)
        identity = input_value
    elif input_type == "view":
        df = spark.table(input_value)
        identity = input_value
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")
    return identity, df


def collect_table_metadata(spark: SparkSession, identity: str, df: DataFrame) -> Dict:
    schema_info = [
        {
            "column_name": f.name,
            "data_type": f.dataType.simpleString(),
            "nullable": f.nullable,
            "metadata": dict(f.metadata) if f.metadata else {},
        }
        for f in df.schema.fields
    ]

    comments: Dict[str, str] = {}
    try:
        db = None
        tbl = None
        if "." in identity and not identity.startswith("/") and not identity.startswith("dbfs:"):
            parts = identity.split(".")
            if len(parts) == 3:
                _, db, tbl = parts
            elif len(parts) == 2:
                db, tbl = parts
        if db and tbl:
            for c in spark.catalog.listColumns(f"{db}.{tbl}"):
                if getattr(c, "comment", None):
                    comments[c.name] = c.comment
    except Exception:
        pass

    return {
        "table_identity": identity,
        "num_columns": len(schema_info),
        "schema": schema_info,
        "column_comments": comments,
    }


# --------------- Profiling ---------------

def profile_table(
    df: DataFrame, sample_fraction: float = 0.25, sample_limit: int = 200_000
) -> DataFrame:
    row_count = df.count()
    if sample_fraction < 1.0 or sample_limit:
        frac = max(0.0001, min(1.0, float(sample_fraction)))
        sampled = df.sample(withReplacement=False, fraction=frac, seed=42)
        if sample_limit:
            sampled = sampled.limit(int(sample_limit))
    else:
        sampled = df
    sampled.cache()
    sampled_count = sampled.count()

    profiles: List[Dict] = []

    email_regex = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    digits_regex = r"^[0-9]+$"
    iso_date_regex = r"^\d{4}-\d{2}-\d{2}$"

    for field in df.schema.fields:
        col_name = field.name
        dtype = field.dataType.simpleString()

        null_count = sampled.filter(F.col(col_name).isNull()).count()
        non_null_df = sampled.filter(F.col(col_name).isNotNull())
        non_null_count = non_null_df.count()
        null_fraction = null_count / max(1, sampled_count)
        approx_distinct = (
            sampled.agg(F.approx_count_distinct(F.col(col_name)).alias("d")).collect()[0]["d"]
        )

        profile: Dict = {
            "column_name": col_name,
            "data_type": dtype,
            "row_count": row_count,
            "sampled_count": sampled_count,
            "null_count": null_count,
            "null_fraction": float(null_fraction),
            "distinct_count": int(approx_distinct) if approx_distinct is not None else None,
        }

        # Numeric
        if dtype.startswith("decimal") or dtype in {"byte", "short", "int", "long", "float", "double"}:
            agg = (
                non_null_df.agg(
                    F.min(F.col(col_name)).alias("min"),
                    F.max(F.col(col_name)).alias("max"),
                    F.mean(F.col(col_name)).alias("mean"),
                    F.stddev(F.col(col_name)).alias("stddev"),
                    F.expr(f"percentile_approx({col_name}, array(0.01,0.5,0.99), 1000)").alias("pct"),
                ).collect()[0]
            )
            p = agg["pct"] if agg["pct"] is not None else [None, None, None]
            profile.update(
                {
                    "min": agg["min"],
                    "max": agg["max"],
                    "mean": agg["mean"],
                    "stddev": agg["stddev"],
                    "p01": p[0],
                    "p50": p[1],
                    "p99": p[2],
                }
            )

        # Date/Timestamp
        elif dtype in {"date", "timestamp"}:
            agg = (
                non_null_df.agg(
                    F.min(F.col(col_name)).alias("min"),
                    F.max(F.col(col_name)).alias("max"),
                    F.expr(
                        f"percentile_approx(unix_timestamp({col_name}), array(0.01,0.5,0.99), 1000)"
                    ).alias("pct"),
                ).collect()[0]
            )
            p = agg["pct"] if agg["pct"] is not None else [None, None, None]
            profile.update(
                {
                    "min": agg["min"],
                    "max": agg["max"],
                    "p01": float(p[0]) if p[0] is not None else None,
                    "p50": float(p[1]) if p[1] is not None else None,
                    "p99": float(p[2]) if p[2] is not None else None,
                }
            )

        # String
        elif dtype == "string":
            lens = non_null_df.select(F.length(F.col(col_name)).alias("len"))
            agg = lens.agg(
                F.min("len").alias("min_len"),
                F.max("len").alias("max_len"),
                F.avg("len").alias("avg_len"),
            ).collect()[0]

            def pattern_fraction(regex: str) -> float:
                return non_null_df.filter(F.col(col_name).rlike(regex)).count() / max(1, non_null_count)

            email_frac = pattern_fraction(email_regex)
            digits_frac = pattern_fraction(digits_regex)
            iso_date_frac = pattern_fraction(iso_date_regex)
            top_vals = (
                non_null_df.groupBy(F.col(col_name)).count().orderBy(F.desc("count")).limit(20)
            )
            top = [r[col_name] for r in top_vals.collect()]
            profile.update(
                {
                    "min_len": agg["min_len"],
                    "max_len": agg["max_len"],
                    "avg_len": float(agg["avg_len"]) if agg["avg_len"] is not None else None,
                    "pattern_email_frac": email_frac,
                    "pattern_digits_frac": digits_frac,
                    "pattern_iso_date_frac": iso_date_frac,
                    "top_values_sample": top,
                }
            )

        profiles.append(profile)

    sampled.unpersist()
    return df.sparkSession.createDataFrame(profiles)


# --------------- Rule recommendation ---------------

def recommend_rules(
    table_identity: str,
    profiles_df: DataFrame,
    table_meta: Dict,
    enum_max_distinct: int = 20,
    enum_max_ratio: float = 0.05,
    unique_tolerance: float = 0.01,
) -> DataFrame:
    profiles = profiles_df.collect()
    comments = table_meta.get("column_comments", {})

    recs: List[Dict] = []
    for p in profiles:
        col = p["column_name"]
        dtype = p["data_type"]
        null_fraction = float(p.get("null_fraction") or 0.0)
        distinct_count = int(p.get("distinct_count") or 0)
        sampled_count = int(p.get("sampled_count") or 1)
        business_desc = comments.get(col)

        # NOT NULL
        if null_fraction == 0.0:
            recs.append(
                {
                    "table": table_identity,
                    "column": col,
                    "rule_type": "NOT_NULL",
                    "rule_expression": f"{col} IS NOT NULL",
                    "parameters": None,
                    "rationale": "No nulls observed in sample",
                    "confidence": 0.85,
                }
            )

        # UNIQUE (approx)
        if null_fraction <= 0.001 and distinct_count >= (1.0 - unique_tolerance) * sampled_count:
            recs.append(
                {
                    "table": table_identity,
                    "column": col,
                    "rule_type": "UNIQUE",
                    "rule_expression": f"approx_count_distinct({col}) == count(*)",
                    "parameters": {"tolerance": unique_tolerance},
                    "rationale": "Distinct count ~ row count; likely key",
                    "confidence": 0.75,
                }
            )

        # ENUM
        if distinct_count and distinct_count <= enum_max_distinct and (distinct_count / max(1, sampled_count)) <= enum_max_ratio:
            values = p.get("top_values_sample")
            if values:
                sample_vals = [v for v in values if v is not None][: min(len(values), enum_max_distinct)]
                if sample_vals:
                    lit_vals = ", ".join([repr(v) for v in sample_vals])
                    recs.append(
                        {
                            "table": table_identity,
                            "column": col,
                            "rule_type": "ENUM",
                            "rule_expression": f"{col} IN ({lit_vals})",
                            "parameters": {"max_distinct": enum_max_distinct},
                            "rationale": "Low cardinality observed; restrict to observed set",
                            "confidence": 0.6,
                        }
                    )

        # RANGE (numeric)
        if dtype.startswith("decimal") or dtype in {"byte", "short", "int", "long", "float", "double"}:
            p01, p99 = p.get("p01"), p.get("p99")
            if p01 is not None and p99 is not None and p99 >= p01:
                recs.append(
                    {
                        "table": table_identity,
                        "column": col,
                        "rule_type": "RANGE",
                        "rule_expression": f"{col} BETWEEN {p01} AND {p99}",
                        "parameters": {"p01": p01, "p99": p99},
                        "rationale": "Numeric range based on central percentiles",
                        "confidence": 0.65,
                    }
                )

        # RANGE (date/timestamp) using epoch second percentiles
        if dtype in {"date", "timestamp"}:
            p01, p99 = p.get("p01"), p.get("p99")
            if p01 is not None and p99 is not None and p99 >= p01:
                recs.append(
                    {
                        "table": table_identity,
                        "column": col,
                        "rule_type": "RANGE",
                        "rule_expression": f"unix_timestamp({col}) BETWEEN {int(p01)} AND {int(p99)}",
                        "parameters": {"p01": int(p01), "p99": int(p99)},
                        "rationale": "Date range based on central percentiles",
                        "confidence": 0.6,
                    }
                )

        # PATTERN
        if dtype == "string":
            email_frac = float(p.get("pattern_email_frac") or 0.0)
            digits_frac = float(p.get("pattern_digits_frac") or 0.0)
            iso_date_frac = float(p.get("pattern_iso_date_frac") or 0.0)
            if email_frac >= 0.8:
                recs.append(
                    {
                        "table": table_identity,
                        "column": col,
                        "rule_type": "PATTERN",
                        "rule_expression": f"{col} RLIKE '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}$'",
                        "parameters": {"min_fraction": 0.8},
                        "rationale": "Most values look like emails",
                        "confidence": 0.7,
                    }
                )
            if digits_frac >= 0.9:
                recs.append(
                    {
                        "table": table_identity,
                        "column": col,
                        "rule_type": "PATTERN",
                        "rule_expression": f"{col} RLIKE '^[0-9]+$'",
                        "parameters": {"min_fraction": 0.9},
                        "rationale": "Mostly digits-only strings",
                        "confidence": 0.6,
                    }
                )
            if iso_date_frac >= 0.6:
                recs.append(
                    {
                        "table": table_identity,
                        "column": col,
                        "rule_type": "PATTERN",
                        "rule_expression": f"{col} RLIKE '^\\\d{{4}}-\\\d{{2}}-\\\d{{2}}$'",
                        "parameters": {"min_fraction": 0.6},
                        "rationale": "Many values resemble ISO date strings",
                        "confidence": 0.55,
                    }
                )

        if business_desc:
            recs.append(
                {
                    "table": table_identity,
                    "column": col,
                    "rule_type": "BUSINESS_CONTEXT",
                    "rule_expression": None,
                    "parameters": {"description": business_desc},
                    "rationale": "Use business description to refine/validate rules",
                    "confidence": 0.5,
                }
            )

    rules_df = profiles_df.sparkSession.createDataFrame(
        recs,
        schema=T.StructType(
            [
                T.StructField("table", T.StringType(), False),
                T.StructField("column", T.StringType(), True),
                T.StructField("rule_type", T.StringType(), False),
                T.StructField("rule_expression", T.StringType(), True),
                T.StructField("parameters", T.MapType(T.StringType(), T.StringType()), True),
                T.StructField("rationale", T.StringType(), True),
                T.StructField("confidence", T.DoubleType(), True),
            ]
        ),
    )
    return rules_df


# --------------- Optional refinements (ML/history) ---------------

def refine_confidence_with_history(
    spark: SparkSession, profiles_df: DataFrame, rules_df: DataFrame, history_path: str | None
) -> DataFrame:
    """Simple heuristic refinement: if a column's profile is stable across history, bump confidence."""
    if not history_path:
        return rules_df

    try:
        hist = spark.read.format("delta").load(history_path)
    except Exception:
        return rules_df

    # Expect history with columns: column_name, metric, value, observed_at
    # We'll compute per-column variance for null_fraction and distinct_count.
    stability = (
        hist.filter(F.col("metric").isin("null_fraction", "distinct_count"))
        .groupBy("column_name", "metric")
        .agg(F.variance("value").alias("var"))
        .groupBy("column_name")
        .agg(F.avg("var").alias("avg_var"))
    )

    enriched = (
        rules_df.alias("r")
        .join(stability.alias("s"), F.col("r.column") == F.col("s.column_name"), "left")
        .withColumn(
            "confidence",
            F.when(F.col("avg_var").isNull(), F.col("confidence")).otherwise(
                F.when(F.col("avg_var") < F.lit(0.001), F.col("confidence") + F.lit(0.05)).otherwise(
                    F.col("confidence")
                )
            ),
        )
        .drop("column_name", "avg_var")
    )
    return enriched


# --------------- Optional LLM hook (stub) ---------------

def llm_enrich_rules_stub(
    rules_df: DataFrame, table_meta: Dict, enable_llm: bool
) -> DataFrame:
    """Placeholder for LLM-based enrichment using business descriptions.

    In practice, call your model endpoint (e.g., Databricks Model Serving or OpenAI/Azure)
    with column business descriptions to generate suggested rule rationales or add new rules.
    This stub just echoes the input when disabled.
    """
    if not enable_llm:
        return rules_df
    # Example of adding a small boost when descriptions exist
    comments = table_meta.get("column_comments", {})
    add_cols = rules_df.withColumn(
        "confidence",
        F.when(F.col("column").isin([*comments.keys()]), F.col("confidence") + F.lit(0.03)).otherwise(
            F.col("confidence")
        ),
    )
    return add_cols


# --------------- Persist rules ---------------

def write_rules_to_delta(rules_df: DataFrame, dq_rules_path: str) -> None:
    with_json = (
        rules_df.select(
            *[c for c in rules_df.columns if c != "parameters"],
            F.to_json(F.col("parameters")).alias("parameters"),
        )
        .withColumn("created_at", F.current_timestamp())
    )
    (
        with_json.write.format("delta").mode("append").option("mergeSchema", "true").save(dq_rules_path)
    )


# --------------- Demo data ---------------

def create_demo_orders(spark: SparkSession, path: str) -> DataFrame:
    data = [
        (1, "alice@example.com", 30, 120.5, "2024-01-10", "PAID"),
        (2, "bob@example.com", 41, 55.0, "2024-02-12", "PAID"),
        (3, "charlie@example", 25, 500.0, "2024-01-15", "PENDING"),
        (4, None, 37, 77.7, "2023-12-30", "CANCELLED"),
        (5, "dave99@company.org", 28, 5.5, "2024-03-01", "PAID"),
        (6, "eve@company.org", None, 88.8, "2024-02-28", "PAID"),
        (7, "frank@example.com", 45, None, "2024-03-05", "PAID"),
        (8, "george77@example.com", 29, 14.2, "2024-02-02", "PAID"),
        (9, "harry@example.com", 32, 9999.9, "2024-02-15", "PAID"),
        (10, "ivan@example.com", 30, 120.5, "2024-01-10", "PAID"),
    ]
    schema = T.StructType(
        [
            T.StructField("order_id", T.IntegerType(), False),
            T.StructField("email", T.StringType(), True),
            T.StructField("age", T.IntegerType(), True),
            T.StructField("amount", T.DoubleType(), True),
            T.StructField("order_date", T.DateType(), True),
            T.StructField("status", T.StringType(), True),
        ]
    )
    df = spark.createDataFrame(data, schema=schema)
    df.write.format("delta").mode("overwrite").save(path)
    return spark.read.format("delta").load(path)


# --------------- Main ---------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-Assisted DQ Rule Recommender (PySpark)")
    p.add_argument("--input-type", required=False, default="view", choices=["table", "path", "view"], help="Input kind")
    p.add_argument("--input-value", required=False, default="demo_orders_view", help="Table name, view name, or Delta path")
    p.add_argument("--dq-rules-path", required=False, default="/tmp/delta/dq_rules", help="Delta path for DQ rules")
    p.add_argument("--sample-fraction", type=float, default=0.25)
    p.add_argument("--sample-limit", type=int, default=200_000)
    p.add_argument("--enum-max-distinct", type=int, default=20)
    p.add_argument("--enum-max-ratio", type=float, default=0.05)
    p.add_argument("--unique-tolerance", type=float, default=0.01)
    p.add_argument("--profiles-history-path", default=None, help="Optional Delta path of historical profiles")
    p.add_argument("--enable-llm", action="store_true", help="Enable LLM enrichment (stub)")
    p.add_argument("--demo", action="store_true", help="Create demo data and temp view, then run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spark = get_spark()
    cfg = DQConfig(
        input_type=args.input_type,
        input_value=args.input_value,
        dq_rules_path=args.dq_rules_path,
        sample_fraction=args.sample_fraction,
        sample_limit=args.sample_limit,
        enum_max_distinct=args.enum_max_distinct,
        enum_max_ratio=args.enum_max_ratio,
        unique_tolerance=args.unique_tolerance,
        profiles_history_path=args.profiles_history_path,
        enable_llm=args.enable_llm,
    )

    if args.demo:
        demo_path = "/tmp/delta/demo_orders"
        create_demo_orders(spark, demo_path).createOrReplaceTempView("demo_orders_view")
        cfg.input_type = "view"
        cfg.input_value = "demo_orders_view"

    identity, df = load_input_df(spark, cfg.input_type, cfg.input_value)
    table_meta = collect_table_metadata(spark, identity, df)
    profiles_df = profile_table(df, cfg.sample_fraction, cfg.sample_limit)
    rules_df = recommend_rules(
        identity,
        profiles_df,
        table_meta,
        enum_max_distinct=cfg.enum_max_distinct,
        enum_max_ratio=cfg.enum_max_ratio,
        unique_tolerance=cfg.unique_tolerance,
    )

    # Optional refinements
    rules_df = refine_confidence_with_history(spark, profiles_df, rules_df, cfg.profiles_history_path)
    rules_df = llm_enrich_rules_stub(rules_df, table_meta, cfg.enable_llm)

    # Persist
    write_rules_to_delta(rules_df, cfg.dq_rules_path)

    # Show preview
    print("Recommended DQ rules (sample):")
    rules_df.orderBy(F.desc("confidence")).show(truncate=False)
    print(f"Saved to: {cfg.dq_rules_path}")


if __name__ == "__main__":
    main()

