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


# -------------------------------
# Improved Schema Serialization
# -------------------------------

def process_tables(database):
    """
    Serialize schema in a structured format:
    table_name(col1, col2, col3)
    + Foreign key info
    Everything lowercased for consistency.
    """

    tables = database["table_names_original"]
    columns = database["column_names_original"]
    foreign_keys = database["foreign_keys"]

    
    table_columns = {i: [] for i in range(len(tables))}

    for table_id, col_name in columns:
        if table_id >= 0:
            table_columns[table_id].append(col_name.lower())

    
    table_strings = []
    for i, table in enumerate(tables):
        table_name = table.lower()
        cols = ", ".join(table_columns[i])
        table_strings.append(f"{table_name}({cols})")

    # Foreign key relationships
    fk_strings = []
    for fk in foreign_keys:
        col1_idx, col2_idx = fk

        t1_id, c1_name = columns[col1_idx]
        t2_id, c2_name = columns[col2_idx]

        table1 = tables[t1_id].lower()
        table2 = tables[t2_id].lower()

        fk_strings.append(
            f"{table1}.{c1_name.lower()} -> {table2}.{c2_name.lower()}"
        )

    schema = "tables:\n" + "\n".join(table_strings)

    if fk_strings:
        schema += "\nforeign_keys:\n" + "\n".join(fk_strings)

    return schema



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