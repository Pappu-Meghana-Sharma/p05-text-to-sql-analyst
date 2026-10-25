# Part 1: Data and SQL Execution Harness

## Dataset
This part uses the Hugging Face dataset [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql).

## Files
- `data.py`: Data loading, train/test splitting, SQL execution, result comparison, and EDA.
- `test_ids.json`: Fixed set of 200 test example IDs for reproducible evaluation.

## Important: SQL Compatibility Limitations

During testing, some gold SQL queries could not execute successfully in SQLite. The observed issues included:

- **Unsupported SQL syntax:** Some queries use syntax from other SQL dialects, such as PostgreSQL `INTERVAL` expressions.
- **Missing tables:** Some queries reference tables that are not defined in their provided `sql_context`.
- **Blocked operations:** Some examples contain `UPDATE` or `DELETE` statements, which the read-only SQL harness intentionally blocks.

These failures may originate from dataset/query compatibility issues rather than the model's SQL generation. Therefore, check the execution error and query context before classifying a failed example as a model error.

The harness intentionally uses read-only execution for safety. These limitations should be considered when interpreting execution accuracy and analysing failure cases.