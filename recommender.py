import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import ast

class ClothingRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.feature_matrix = None
        self.index = None
        
        # Define which brands should have metadata processed
        self.process_metadata_brands = {'Zara', 'H&M'}
        
    def load_data(self):
        """Load and preprocess the product data"""
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} products")
        
    def _process_metadata(self, metadata_str, brand):
        """Process metadata string based on brand"""
        if not isinstance(metadata_str, str) or brand not in self.process_metadata_brands:
            return {}
            
        try:
            # Try parsing as JSON first
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            try:
                # Try evaluating as literal Python dict
                metadata = ast.literal_eval(metadata_str)
            except (ValueError, SyntaxError):
                print(f"Error processing metadata for {brand}")
                return {}
                
        return metadata if isinstance(metadata, dict) else {}
        
    def _extract_features(self):
        """Extract features from product data"""
        # Combine text features
        combined_features = []
        
        for _, row in self.data.iterrows():
            features = []
            
            # Add product name
            if pd.notna(row.get('product_name')):
                features.append(str(row['product_name']))
                
            # Add description
            if pd.notna(row.get('description')):
                features.append(str(row['description']))
                
            # Process metadata based on brand
            if pd.notna(row.get('metadata')) and pd.notna(row.get('brand')):
                metadata = self._process_metadata(row['metadata'], row['brand'])
                if metadata:
                    # Add metadata values to features, handling numpy arrays
                    for v in metadata.values():
                        if isinstance(v, (str, int, float)) and not pd.isna(v):
                            features.append(str(v))
            
            combined_features.append(' '.join(features))
            
        # Create TF-IDF matrix
        self.feature_matrix = self.vectorizer.fit_transform(combined_features)
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
    def build_index(self):
        """Build Faiss index for fast similarity search"""
        # Convert sparse matrix to dense and normalize
        dense_matrix = self.feature_matrix.toarray().astype('float32')
        faiss.normalize_L2(dense_matrix)
        
        # Build index
        self.index = faiss.IndexFlatIP(dense_matrix.shape[1])
        self.index.add(dense_matrix)
        print("Faiss index built successfully")
        
    def recommend(self, query, n=5):
        """Get recommendations based on query text
        
        Args:
            query (str): Query text to find similar products
            n (int): Number of recommendations to return
            
        Returns:
            List of tuples (index, similarity_score)
        """
        # Transform query
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self.index.search(query_vector, n)
        
        # Return list of (index, score) tuples
        return list(zip(indices[0], scores[0]))

def main():
    # Initialize the recommender
    recommender = ClothingRecommender('./preprocessed_products.csv')
    
    # Load data and build index
    recommender.load_data()
    recommender._extract_features()
    recommender.build_index()
    
    # Example recommendation
    query = "blue shirt"
    recommendations = recommender.recommend(query, n=20)
    
    print(f"\nTop 20 recommendations for '{query}':")
    for i, (idx, score) in enumerate(recommendations, 1):
        product = recommender.data.iloc[idx]
        print(f"\n{i}. {product['product_name']} ({product['brand']})")
        print(f"   Score: {score:.4f}")
        print(f"   Price: {product['product_price']}")
        print(f"   URL: {product['product_url']}")

if __name__ == "__main__":
    main() 