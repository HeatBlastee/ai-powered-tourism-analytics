import pandas as pd
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Configure structured logging
logger = logging.getLogger(__name__)

class TourismRecommender:
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.place_ids: Optional[List[int]] = None
        self.interaction_matrix: Optional[pd.DataFrame] = None
        
    def fit(self, interaction_matrix: pd.DataFrame) -> None:
        """
        Fits the recommender by building a FAISS index.
        interaction_matrix: DataFrame with index=User_Id, columns=Place_Id
        """
        logger.info(f"Fitting Recommender on matrix shape: {interaction_matrix.shape}")
        self.interaction_matrix = interaction_matrix
        
        # Create user-item matrix (Transposed interaction matrix)
        # Rows = Places, Cols = Users
        item_user_matrix = self.interaction_matrix.T
        
        # Convert to float32 numpy array for FAISS
        vectors = item_user_matrix.values.astype('float32')
        self.place_ids = list(item_user_matrix.index)
        
        # Normalize vectors for Cosine Similarity (L2 normalization)
        faiss.normalize_L2(vectors)
        
        # Build Index (Inner Product)
        # Dimensions = number of users
        d = vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors)
        logger.info(f"FAISS Index built with {self.index.ntotal} items.")
        
    def recommend(self, place_name: str, places_df: pd.DataFrame, top_n: int = 5) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Recommends places similar to the given place_name using FAISS.
        """
        if self.index is None:
            logger.error("Attempted to recommend without fitting the model.")
            raise RuntimeError("Model not fit yet. Call fit() first.")
            
        # Find Place_Id for the name
        try:
            place_row = places_df[places_df['Place_Name'] == place_name]
            if place_row.empty:
                logger.warning(f"Place '{place_name}' not found in database.")
                return None, f"Place '{place_name}' not found."
            place_id = place_row['Place_Id'].values[0]
        except IndexError:
            return None, f"Place '{place_name}' not found."
            
        if self.place_ids is None or place_id not in self.place_ids:
            logger.warning(f"No interaction data for place_id={place_id} ({place_name}).")
            return None, f"No rating data available for '{place_name}'."
            
        # Get index of the place in our FAISS array
        # No need for internal_idx logic since we query by vector
        
        # Retrieve vector and query index
        try:
            query_vector = self.interaction_matrix.T.loc[place_id].values.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search (k = top_n + 1 because the first result is the item itself)
            distances, indices = self.index.search(query_vector, top_n + 1)
            
            # Process results
            recommendations = []
            for i, idx in enumerate(indices[0]):
                neighbor_place_id = self.place_ids[idx]
                
                # Skip the place itself
                if neighbor_place_id == place_id:
                    continue
                    
                score = distances[0][i]
                
                # Get details
                place_details = places_df[places_df['Place_Id'] == neighbor_place_id].iloc[0].to_dict()
                place_details['Similarity'] = float(score)
                recommendations.append(place_details)
                
            return pd.DataFrame(recommendations).head(top_n), None
            
        except Exception as e:
            logger.error(f"Recommendation failed for {place_name}: {e}")
            raise
        
    def save(self, path: Any) -> None:
        # Save FAISS index
        path = Path(path)
        if self.index:
            faiss.write_index(self.index, str(path))
            logger.info(f"FAISS index saved to {path}")
        
        # Save metadata (place_ids mapping)
        metadata_path = path.parent / f"{path.stem}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({'place_ids': self.place_ids, 'interaction_matrix': self.interaction_matrix}, f)
        logger.info(f"Metadata saved to {metadata_path}")
            
    def load(self, path: Any) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
            
        self.index = faiss.read_index(str(path))
        
        metadata_path = path.parent / f"{path.stem}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.place_ids = data['place_ids']
            self.interaction_matrix = data['interaction_matrix']
            
        logger.info(f"FAISS index loaded from {path}")
