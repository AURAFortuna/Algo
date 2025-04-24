import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from typing import List, Dict

class SimpleRecommender:
    def __init__(self):
        """Initialize the SVD recommender."""
        self.model = SVD()
        self.user_interactions = None

    def load_data(self, file_path: str):
        """Load and prepare the data."""
        # Read the CSV file
        df = pd.read_csv(file_path)
        self.user_interactions = df
        
        # Define the rating scale
        reader = Reader(rating_scale=(-4, 3))
        
        # Load the data into Surprise format
        data = Dataset.load_from_df(df[['user_id', 'outfit_id', 'interaction']], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
        return df

    def recommend_for_user(self, user_id: str = "USER_001", n_recommendations: int = 5):
        """Generate recommendations for a specific user."""
        # Get all unique outfit IDs
        all_outfits = set(self.user_interactions['outfit_id'])
        
        # Get outfits the user has already interacted with and their ratings
        user_rated_outfits = self.user_interactions[
            self.user_interactions['user_id'] == user_id
        ][['outfit_id', 'interaction']].set_index('outfit_id')['interaction'].to_dict()
        
        # Get predictions for all outfits with penalty for previously rated ones
        predictions = []
        for outfit_id in all_outfits:
            pred = self.model.predict(user_id, outfit_id)
            # Apply penalty if outfit was previously rated
            if outfit_id in user_rated_outfits:
                # Penalize based on the original rating:
                # - For disliked outfits (-1), reduce score by 3.0
                # - For liked outfits (1), reduce score by 3.0
                # - For wishlisted outfits (3), reduce score by 3.5
                penalty = {
                    -1: 3.0,
                    1: 3.0,
                    3: 3.5
                }.get(user_rated_outfits[outfit_id], 1.0)
                adjusted_score = pred.est - penalty
            else:
                adjusted_score = pred.est
            
            predictions.append((outfit_id, adjusted_score, pred.est))
        
        # Sort by adjusted score and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:n_recommendations]
        
        # Print recommendations
        print(f"\nTop {n_recommendations} recommendations for {user_id}:")
        for outfit_id, adjusted_score, original_score in top_n:
            status = "Previously rated" if outfit_id in user_rated_outfits else "New"
            print(f"Outfit: {outfit_id}, Adjusted Score: {adjusted_score:.2f}, Original Score: {original_score:.2f} ({status})")

def main():
    # Initialize the recommender
    recommender = SimpleRecommender()
    
    # Load data and train model
    print("Loading data and training model...")
    recommender.load_data('small_user_outfit_interactions.csv')
    
    # Generate recommendations for USER_001
    recommender.recommend_for_user()

if __name__ == "__main__":
    main() 