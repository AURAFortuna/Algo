import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import numpy as np
import os
import json
import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration via Environment Variables ---
INTERACTIONS_FILE_PATH = os.environ.get('INTERACTIONS_FILE_PATH', 'data/sample_user_outfit_interactions.parquet')
MIN_INTERACTIONS_PER_OUTFIT = int(os.environ.get('MIN_INTERACTIONS_PER_OUTFIT', 5))
TOP_K_SIMILAR = int(os.environ.get('TOP_K_SIMILAR', 50))

def load_and_preprocess_data_from_local(filepath, min_interactions):
    logging.info(f"Loading data from local file: {filepath}")
    
    try:
        interactions_df = pd.read_parquet(filepath)
        if not {'user_id', 'outfit_id', 'interaction'}.issubset(interactions_df.columns):
             raise ValueError("Parquet file must contain 'user_id', 'outfit_id', and 'interaction' columns.")
        logging.info(f"Successfully loaded {len(interactions_df)} initial interactions from local file.")
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame(), {}

    # --- Filtering ---
    logging.info("\nInteraction distribution:")
    logging.info(interactions_df['interaction'].value_counts().sort_index())
    outfit_counts = interactions_df['outfit_id'].value_counts()
    outfits_to_keep = outfit_counts[outfit_counts >= min_interactions].index
    interactions_df = interactions_df[interactions_df['outfit_id'].isin(outfits_to_keep)]
    logging.info(f"\nInteractions after filtering outfits: {len(interactions_df)}")
    logging.info(f"Number of outfits after filtering: {len(outfits_to_keep)}")

    if interactions_df.empty:
        logging.warning("DataFrame is empty after filtering. No data to process.")
        return pd.DataFrame(), {}

    # Create mappings
    interactions_df['user_idx'] = interactions_df['user_id'].astype('category').cat.codes
    interactions_df['outfit_idx'] = interactions_df['outfit_id'].astype('category').cat.codes

    # Create index to original ID mapping
    outfit_mapping_df = interactions_df[['outfit_idx', 'outfit_id']].drop_duplicates().set_index('outfit_idx')
    outfit_idx_to_original_id = outfit_mapping_df['outfit_id'].to_dict()

    logging.info(f"\nNumber of unique users: {interactions_df['user_idx'].nunique()}")
    logging.info(f"Number of unique outfits: {interactions_df['outfit_idx'].nunique()}")

    return interactions_df, outfit_idx_to_original_id

def create_sparse_user_outfit_matrix(df):
    logging.info("\nCreating sparse user-outfit matrix...")
    num_users = df['user_idx'].nunique()
    num_outfits = df['outfit_idx'].nunique()
    sparse_matrix = csr_matrix((df['interaction'], (df['user_idx'], df['outfit_idx'])),
                             shape=(num_users, num_outfits))
    return sparse_matrix

def calculate_outfit_similarity(sparse_matrix, k=TOP_K_SIMILAR, outfit_idx_map=None):
    logging.info(f"\nExtracting top {k} similar outfits for each outfit...")
    start_time = time.time()
    
    # Calculate cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(sparse_matrix.T)
    logging.info(f"Calculated cosine similarity matrix of shape: {cosine_sim_matrix.shape}")
    
    # --- Corrected Population ---
    top_k_similarities = defaultdict(list)
    # Convert numpy array to COO format
    cosine_sim_coo = csr_matrix(cosine_sim_matrix).tocoo()
    outfit_similarities = defaultdict(list)
    
    # Group similarities by outfit index 'i'
    for i, j, v in zip(cosine_sim_coo.row, cosine_sim_coo.col, cosine_sim_coo.data):
        if i != j:  # Exclude self-similarity
            outfit_similarities[i].append((j, v))

    processed_count = 0
    for outfit_idx, neighbors in outfit_similarities.items():
        if outfit_idx not in outfit_idx_map:
            continue

        neighbors.sort(key=lambda x: x[1], reverse=True)
        top_k = neighbors[:k]

        original_outfit_id = outfit_idx_map[outfit_idx]
        original_neighbors = []
        for neighbor_idx, similarity in top_k:
            if neighbor_idx in outfit_idx_map:
                original_neighbor_id = outfit_idx_map[neighbor_idx]
                original_neighbors.append([original_neighbor_id, float(similarity)])

        if original_neighbors:
            top_k_similarities[original_outfit_id] = original_neighbors
            processed_count += 1

    logging.info(f"Top-K extraction took {time.time() - start_time:.2f} seconds.")
    logging.info(f"Generated similarities for {processed_count} outfits")
    return dict(top_k_similarities)

def save_similarities_to_csv(similarities_dict, output_file='outfit_similarities.csv'):
    """Saves the computed similarities dictionary to CSV."""
    logging.info(f"\nSaving similarities to {output_file}...")
    try:
        # Convert dictionary to DataFrame
        rows = []
        for outfit_id, neighbors in similarities_dict.items():
            for neighbor_id, similarity in neighbors:
                rows.append({
                    'outfit_id': outfit_id,
                    'similar_outfit_id': neighbor_id,
                    'similarity_score': similarity
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logging.info(f"Successfully saved {len(rows)} similarity pairs to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        raise

if __name__ == "__main__":
    logging.info("Starting outfit similarity calculation job...")

    # Call the function
    result = load_and_preprocess_data_from_local(
        INTERACTIONS_FILE_PATH, MIN_INTERACTIONS_PER_OUTFIT
    )

    # Check if the result is valid before unpacking
    if result is not None and isinstance(result, tuple) and len(result) == 2:
        interactions, outfit_mapping = result
        if interactions.empty:
             logging.warning("Received empty interactions DataFrame after loading/preprocessing. Exiting job.")
        elif not outfit_mapping:
             logging.warning("Received empty outfit mapping after loading/preprocessing. Exiting job.")
        else:
            # --- Proceed with calculations ONLY if data is valid ---
            user_outfit_sparse = create_sparse_user_outfit_matrix(interactions)
            # Check if outfit_mapping is necessary and valid before passing
            if outfit_mapping:
                 top_k_similar_outfits = calculate_outfit_similarity(
                      user_outfit_sparse, k=TOP_K_SIMILAR, outfit_idx_map=outfit_mapping
                 )

                 if top_k_similar_outfits:
                      save_similarities_to_csv(top_k_similar_outfits)
                      logging.info("\nOffline similarity calculation and CSV update complete.")
                 else:
                      logging.info("\nNo similarities generated to save.")
            else:
                 logging.error("Outfit mapping is missing or invalid after preprocessing. Cannot calculate similarities.")
    else:
        logging.error(f"Failed to load or preprocess data correctly. Function returned: {result}. Exiting job.") 