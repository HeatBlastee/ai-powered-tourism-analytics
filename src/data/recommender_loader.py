import pandas as pd
import numpy as np
from pathlib import Path
from src.settings import settings
from src.data.mock_generator import MockDataGenerator

class RecommenderDataLoader:
    def __init__(self, data_dir: Path = settings.RECOMMENDER_DATA_DIR):
        self.data_dir = data_dir
        
        # Check if data exists
        required_files = ["tourism_rating.csv", "user.csv", "tourism_with_id.xlsx"]
        missing = [f for f in required_files if not (self.data_dir / f).exists()]
        
        if missing:
            print(f"Missing files: {missing}. Generating mock data...")
            MockDataGenerator.generate_recommender_data()

    def load_data(self):
        """
        Loads user, rating, and place data.
        Returns:
            ratings_df, users_df, places_df
        """
        try:
            ratings = pd.read_csv(self.data_dir / "tourism_rating.csv")
            users = pd.read_csv(self.data_dir / "user.csv")
            places = pd.read_excel(self.data_dir / "tourism_with_id.xlsx")
            
            # Basic cleaning
            ratings.dropna(inplace=True)
            
            return ratings, users, places
            
        except Exception as e:
            raise RuntimeError(f"Error loading recommender data: {str(e)}")

    def get_interaction_matrix(self):
        """
        Creates a User-Item interaction matrix.
        """
        ratings, _, _ = self.load_data()
        
        # Create pivot table
        matrix = ratings.pivot_table(
            index="User_Id", 
            columns="Place_Id", 
            values="Place_Ratings"
        ).fillna(0)
        
        return matrix
