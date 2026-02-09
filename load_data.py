import json
from pathlib import Path

class Spider:
    def __init__(self, data_path):
        self.data_path = Path(data_path)

        self.train_path = self.data_path / 'train_spider.json'
        self.test_path = self.data_path / 'dev.json'
        self.tables_path = self.data_path / 'tables.json'

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

    cols = {}
    for i in range(len(tables)):
        cols[i] = []

    for id, name in columns:
        if id >= 0:
            cols[id].append(name)

    table_info = []
    for i, table in enumerate(tables):
        col = ", ".join(cols[i])
        table_info.append(f"{table}({col})")
    
    return " ; ".join(table_info)


def process_query(data, database):
    inputs = []
    results = []

    for example in data:
        question = example["question"]
        query = example["query"]
        id = example["db_id"]

        table = process_tables(database[id])
        string = (
            "translate to SQL: "
            f"question: {question}"
            f"table: {table}"
        )

        inputs.append(string)
        results.append(query)
    return inputs, results
