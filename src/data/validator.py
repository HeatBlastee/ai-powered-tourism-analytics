import pandas as pd
import mlflow
from typing import Tuple, Dict, Any

class DataValidator:
    def __init__(self):
        self.validation_report = {}

    def validate_recommender_data(
        self, 
        ratings: pd.DataFrame, 
        users: pd.DataFrame, 
        places: pd.DataFrame
    ) -> bool:
        """
        Validates Recommender System Data.
        Returns True if passed, raises ValueError if failed heavily.
        """
        print("Running Data Validation...")
        
        # 1. Schema Validation
        self._check_columns(ratings, ['User_Id', 'Place_Id', 'Place_Ratings'], 'ratings')
        self._check_columns(users, ['User_Id', 'Age', 'Location'], 'users')
        self._check_columns(places, ['Place_Id', 'Place_Name', 'Category'], 'places')
        
        # 2. Null Value Check
        self._check_nulls(ratings, threshold=0.05, name='ratings') # Allow max 5% nulls
        self._check_nulls(places, threshold=0.10, name='places')   # Allow max 10% nulls
        
        # 3. Duplicate Check
        self._check_duplicates(ratings, subset=['User_Id', 'Place_Id'], name='ratings')
        
        # 4. User/Item Overlap
        self._check_overlap(ratings, users, places)
        
        # Log Report to MLflow
        mlflow.log_dict(self.validation_report, "data_validation_report.json")
        print("Data Validation Passed! Report logged to MLflow.")
        return True

    def _check_columns(self, df: pd.DataFrame, required: list, name: str):
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"[{name}] Missing critical columns: {missing}")
        self.validation_report[f"{name}_schema"] = "PASSED"

    def _check_nulls(self, df: pd.DataFrame, threshold: float, name: str):
        null_pct = df.isnull().mean().max()
        if null_pct > threshold:
            raise ValueError(f"[{name}] Null rate {null_pct:.2%} exceeds threshold {threshold:.2%}")
        self.validation_report[f"{name}_null_check"] = f"PASSED ({null_pct:.2%})"

    def _check_duplicates(self, df: pd.DataFrame, subset: list, name: str):
        dupes = df.duplicated(subset=subset).sum()
        if dupes > 0:
            print(f"Warning: [{name}] Found {dupes} duplicate entries. Dropping them.")
            df.drop_duplicates(subset=subset, inplace=True)
            self.validation_report[f"{name}_duplicates"] = f"FIXED (Removed {dupes})"
        else:
            self.validation_report[f"{name}_duplicates"] = "PASSED"

    def _check_overlap(self, ratings, users, places):
        # Users in ratings vs Users in user table
        valid_users = ratings['User_Id'].isin(users['User_Id']).mean()
        valid_places = ratings['Place_Id'].isin(places['Place_Id']).mean()
        
        self.validation_report['user_coverage'] = f"{valid_users:.2%}"
        self.validation_report['place_coverage'] = f"{valid_places:.2%}"
        
        if valid_users < 0.8:
            print("Warning: < 80% of rating users found in user table.")
