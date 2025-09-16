 AI‑Assisted Data Quality (DQ) Rule Recommender for Delta Tables
 
 Overview
 - Profiles Delta tables and recommends DQ rules using heuristics.
 - Optional ML/history refinement and optional LLM enrichment based on business descriptions.
 - Persists rules to a Delta DQ table with rationale and confidence.
 
 Quick Demo (Databricks)
 1. Open the notebook at `notebooks/ai_dq_recommender.ipynb` and run all cells.
 2. Or run the script as a job with `--demo` to create sample data:
    - `python ai_dq_recommender.py --input-type view --input-value demo_orders_view --dq-rules-path dbfs:/tmp/delta/dq_rules --demo`
 
 Quick Demo (OSS Spark)
 - Use spark-submit with Delta extensions (adjust version as needed):
   - `spark-submit --packages io.delta:delta-spark_2.12:3.2.0 \
      --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
      --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
      ai_dq_recommender.py --input-type path --input-value /tmp/delta/demo_orders --dq-rules-path /tmp/delta/dq_rules --demo`
 
 Inputs
 - Table identity via `--input-type` and `--input-value`:
   - `table`: fully qualified catalog table (e.g., `hive_metastore.db.orders`)
   - `path`: Delta path (e.g., `dbfs:/mnt/.../delta/orders` or `/tmp/delta/orders`)
   - `view`: a temporary view in the session
 
 Outputs
 - Delta path containing recommended rules with schema:
   - `table` (STRING), `column` (STRING), `rule_type` (STRING), `rule_expression` (STRING),
   - `parameters` (JSON string), `rationale` (STRING), `confidence` (DOUBLE), `created_at` (TIMESTAMP)
 
 Heuristic Rules
 - NOT NULL: when null_fraction == 0 in sample
 - UNIQUE (approx): when distinct_count ≈ sampled_count and few nulls
 - ENUM: low-cardinality strings suggest an IN-list
 - RANGE: numeric/date window using p01–p99 percentiles
 - PATTERN: common patterns like email/digits/ISO date
 
 Optional Enhancements
 - History refinement: bump confidence for stable columns across time
 - LLM hook: use business descriptions (catalog comments) to enrich rationale
 
 PPT Notes (1 slide each)
 1) Problem: Manual DQ rule authoring is slow and stale.
 2) Approach: Profile + Heuristics + Optional ML/LLM.
 3) Architecture: Delta table → Profiling → Recommender → DQ Delta sink.
 4) Demo: Run notebook/script, inspect suggested rules, and discuss next steps.
 5) Next: Connect lineage, orchestrate daily refresh, add human-in-the-loop review.
 
 Databricks Model Serving Integration
 - Purpose: Use your hosted LLM to enrich rules with business descriptions.
 - Requirements:
   - A Serving Endpoint that accepts chat-style payloads (JSON with `messages`).
   - `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars set, or pass flags.
 - Create a PAT: User Settings → Developer → Access Tokens.
 - Example run:
   - `python ai_dq_recommender.py --demo --enable-llm \
      --dbx-endpoint my-llm-endpoint \
      --dbx-host https://<your-workspace-host> \
      --dbx-token <PAT> \
      --dq-rules-path dbfs:/tmp/delta/dq_rules`
 - What the model receives:
   - `system` prompt describing rule types and required JSON output.
   - `user` JSON: table identity, column comments (business desc), compact profile summaries.
 - What the model returns:
   - JSON array: `{column, rule_type, rule_expression, rationale, confidence}`.
   - Script validates and unions as `LLM_<rule_type>` with clipped confidence.

