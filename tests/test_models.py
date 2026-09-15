import pytest
import numpy as np
import pandas as pd
from src.models.heritage_classifier import HeritageClassifier
from src.models.tourism_recommender import TourismRecommender

def test_heritage_classifier_build():
    classifier = HeritageClassifier(num_classes=5)
    assert classifier.model is not None
    # Check output shape
    assert classifier.model.output_shape == (None, 5)

def test_recommender_fit():
    recommender = TourismRecommender()
    
    # Create dummy interaction matrix
    data = {
        'Place_1': [5, 0, 4],
        'Place_2': [0, 5, 0],
        'Place_3': [2, 0, 5]
    }
    df = pd.DataFrame(data, index=['User_1', 'User_2', 'User_3'])
    
    recommender.fit(df)
    
    assert recommender.index is not None
    assert recommender.index.ntotal == 3  # 3 places (items) indexed
