import mlflow
import time
from src.data.recommender_loader import RecommenderDataLoader
from src.models.tourism_recommender import TourismRecommender
from src.data.validator import DataValidator
from src.config import RECOMMENDER_MODEL_PATH

def main():
    print("Starting Recommender Model Training...")
    mlflow.set_experiment("Tourism Recommender")
    
    with mlflow.start_run():
        # Load Data
        loader = RecommenderDataLoader()
        ratings, users, places = loader.load_data()
        
        # --- DATA VALIDATION (New Step) ---
        validator = DataValidator()
        validator.validate_recommender_data(ratings, users, places)
        # ----------------------------------
        
        # Create Matrix after validation/cleaning
        interaction_matrix = ratings.pivot_table(
            index="User_Id", 
            columns="Place_Id", 
            values="Place_Ratings"
        ).fillna(0)
        
        n_users, n_items = interaction_matrix.shape
        mlflow.log_param("n_users", n_users)
        mlflow.log_param("n_items", n_items)
        
        # Build & Fit Model
        start_time = time.time()
        recommender = TourismRecommender()
        recommender.fit(interaction_matrix)
        build_time = time.time() - start_time
        
        mlflow.log_metric("build_time_seconds", build_time)
        print(f"Index built in {build_time:.2f} seconds")
        
        # Save Model
        recommender.save(RECOMMENDER_MODEL_PATH)
        mlflow.log_artifact(str(RECOMMENDER_MODEL_PATH))
    
    print("Training finished.")

if __name__ == "__main__":
    main()
