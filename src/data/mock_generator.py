import numpy as np
import pandas as pd
import cv2
import os
from src.config import VISION_DATA_DIR, RECOMMENDER_DATA_DIR

class MockDataGenerator:
    """
    Generates mock data for heritage classification and tourism recommendation
    if real data is missing.
    """
    
    @staticmethod
    def generate_vision_data(num_classes=3, images_per_class=10):
        """Generates dummy images for classification testing."""
        print(f"Generating mock vision data in {VISION_DATA_DIR}...")
        
        classes = [f"heritage_site_{i}" for i in range(num_classes)]
        
        for cls in classes:
            cls_dir = VISION_DATA_DIR / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(images_per_class):
                # Create a random color image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                # Add some text to make it distinguishable
                cv2.putText(img, f"{cls}_{i}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imwrite(str(cls_dir / f"mock_{i}.jpg"), img)
                
        print("Mock vision data generated.")

    @staticmethod
    def generate_recommender_data(num_users=100, num_places=50):
        """Generates dummy CSV files for recommender system."""
        print(f"Generating mock recommender data in {RECOMMENDER_DATA_DIR}...")
        
        RECOMMENDER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # unique places
        place_ids = range(1, num_places + 1)
        places = pd.DataFrame({
            'Place_Id': place_ids,
            'Place_Name': [f"Place {i}" for i in place_ids],
            'Category': np.random.choice(['Nature', 'History', 'Art'], num_places),
            'Rating': np.random.uniform(3.0, 5.0, num_places).round(1)
        })
        places.to_csv(RECOMMENDER_DATA_DIR / "tourism_rating.csv", index=False) # Changed from excel to csv for simplicity
        
        # users
        user_ids = range(1, num_users + 1)
        users = pd.DataFrame({
            'User_Id': user_ids,
            'Age': np.random.randint(18, 60, num_users),
            'Location': np.random.choice(['New York', 'London', 'Paris', 'Tokyo'], num_users)
        })
        users.to_csv(RECOMMENDER_DATA_DIR / "user.csv", index=False)
        
        # ratings
        ratings = []
        for uid in user_ids:
            # Each user rates 3-10 places
            rated_places = np.random.choice(place_ids, np.random.randint(3, 10), replace=False)
            for pid in rated_places:
                ratings.append({
                    'User_Id': uid,
                    'Place_Id': pid,
                    'Place_Ratings': np.random.randint(1, 6)
                })
        
        ratings_df = pd.DataFrame(ratings)
        ratings_df.to_csv(RECOMMENDER_DATA_DIR / "tourism_rating.csv", index=False) # Wait, typically separated. 
        # Actually in the original, `tourism_with_id.xlsx` had place info, `user.csv` user info, `tourism_rating.csv` ratings.
        # I'll overwrite to match:
        
        places.to_excel(RECOMMENDER_DATA_DIR / "tourism_with_id.xlsx", index=False)
        ratings_df.to_csv(RECOMMENDER_DATA_DIR / "tourism_rating.csv", index=False)
        
        print("Mock recommender data generated.")
