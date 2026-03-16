import json
from pathlib import Path


class Spider:
    def __init__(self, data_path):
        self.data_path = Path(data_path)

        self.train_path = self.data_path / "train_spider.json"
        self.test_path = self.data_path / "dev.json"
        self.tables_path = self.data_path / "tables.json"

        self.train = self.load_json(self.train_path)
        self.test = self.load_json(self.test_path)
        self.tables = self.load_json(self.tables_path)

    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def load_tables(self, path):
        databases = {}
        with open(path, "r") as f:
            data = json.load(f)
            for database in data:
                databases[database["db_id"]] = database
        return databases


def process_tables(database):

    tables = database["table_names_original"]
    columns = database["column_names_original"]
    foreign_keys = database["foreign_keys"]

    # map table_id -> list of columns
    table_columns = {i: [] for i in range(len(tables))}

    for table_id, col_name in columns:
        if table_id >= 0:
            table_columns[table_id].append(col_name.lower())

    schema_parts = []
    schema_parts.append("Database Schema:\n")

    for table_id, table_name in enumerate(tables):
        table_name = table_name.lower()
        schema_parts.append(f"Table: {table_name}")

        for col in table_columns[table_id]:
            schema_parts.append(f"  - {col}")

        schema_parts.append("")

    # deal with foreign keys
    if foreign_keys:
        schema_parts.append("Foreign Keys:")

        for col1_idx, col2_idx in foreign_keys:
            t1_id, c1_name = columns[col1_idx]
            t2_id, c2_name = columns[col2_idx]

            table1 = tables[t1_id].lower()
            table2 = tables[t2_id].lower()

            schema_parts.append(
                f"  - {table1}.{c1_name.lower()} = {table2}.{c2_name.lower()}"
            )

    return "\n".join(schema_parts)



def build_prompt(question, schema):
    """
    Stronger instruction-based prompt.
    """

    return (
        "You are a SQLite expert.\n"
        "Given a question and database schema, generate a valid SQLite query.\n"
        "Only output the SQL query.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question:\n{question.lower()}\n\n"
        "SQL:\n"
    )



def process_query(data, database):
    """
    Build training inputs and targets.
    Lowercase everything for consistency.
    """

    inputs = []
    targets = []

    for example in data:
        question = example["question"].lower()
        query = example["query"].lower()
        db_id = example["db_id"]

        schema = process_tables(database[db_id])
        prompt = build_prompt(question, schema)

        inputs.append(prompt)
        targets.append(query)

    return inputs, targets