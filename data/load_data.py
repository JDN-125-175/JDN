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