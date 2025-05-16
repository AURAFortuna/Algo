import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import numpy as np
import os
import json # Still use JSON for serialization within Firestore document
import logging

# --- GCP Client Libraries ---
from google.cloud import storage
from google.cloud import firestore # Import Firestore

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration via Environment Variables ---
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'outfit-ratings-bucket')
INTERACTIONS_FILE_GCS_PATH = os.environ.get('INTERACTIONS_FILE_GCS_PATH', 'sample_user_outfit_interactions.parquet')
# REDIS environment variables no longer needed for this script
FIRESTORE_COLLECTION_NAME = os.environ.get('FIRESTORE_COLLECTION_NAME', 'outfit_similarities') # Firestore collection

MIN_INTERACTIONS_PER_OUTFIT = int(os.environ.get('MIN_INTERACTIONS_PER_OUTFIT', 5))
TOP_K_SIMILAR = int(os.environ.get('TOP_K_SIMILAR', 50))

# --- Functions (load_and_preprocess, create_sparse, calculate_similarity remain largely the same) ---
def load_and_preprocess_data_from_gcs(bucket_name, gcs_filepath, min_interactions):
    # Construct GCS URI
    gcs_uri = f"gs://{bucket_name}/{gcs_filepath}"
    logging.info(f"Loading data from GCS: {gcs_uri}")

    try:
        interactions_df = pd.read_parquet(gcs_uri)
        if not {'user_id', 'outfit_id', 'interaction'}.issubset(interactions_df.columns):
             raise ValueError("Parquet file must contain 'user_id', 'outfit_id', and 'interaction' columns.")
        logging.info(f"Successfully loaded {len(interactions_df)} initial interactions from GCS.")
    except Exception as e:
        logging.error(f"Error loading data from GCS {gcs_uri}: {e}")
        # Return empty structures, not None
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
        # Return empty structures, not None
        return pd.DataFrame(), {}
    # --- End Filtering ---

    # --- !!! ADD THESE LINES BACK !!! ---
    # Create mappings (now uses the filtered DataFrame)
    interactions_df['user_idx'] = interactions_df['user_id'].astype('category').cat.codes
    # Create outfit index
    interactions_df['outfit_idx'] = interactions_df['outfit_id'].astype('category').cat.codes

    # Create index to original ID mapping using filtered data
    outfit_mapping_df = interactions_df[['outfit_idx', 'outfit_id']].drop_duplicates().set_index('outfit_idx')
    outfit_idx_to_original_id = outfit_mapping_df['outfit_id'].to_dict()
    # --- END OF ADDED LINES ---

    logging.info(f"\nNumber of unique users: {interactions_df['user_idx'].nunique()}")
    # Use .nunique() on the index column for accuracy after mapping
    logging.info(f"Number of unique outfits: {interactions_df['outfit_idx'].nunique()}")

    # Ensure you return the calculated mapping
    return interactions_df, outfit_idx_to_original_id

def create_sparse_user_outfit_matrix(df):
    # ... (Keep the existing logic) ...
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
            # ... (warning log) ...
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
            # THIS IS THE KEY PART: Assign to the correct dictionary
            top_k_similarities[original_outfit_id] = original_neighbors
            processed_count += 1
    # --- End Correction ---

    logging.info(f"Top-K extraction took {time.time() - start_time:.2f} seconds.")
    logging.info(f"Generated similarities for {processed_count} outfits")
    return dict(top_k_similarities) # Return the populated dictionary


def save_similarities_to_firestore(similarities_dict, collection_name):
    """Saves the computed similarities dictionary to Firestore."""
    logging.info(f"\nConnecting to Firestore and saving to collection '{collection_name}'...")
    try:
        # Initialize Firestore client. ADC will be used automatically in GCP.
        db = firestore.Client()
        logging.info("Successfully connected to Firestore.")
    except Exception as e:
        logging.error(f"Failed to connect to Firestore: {e}")
        raise

    logging.info(f"Preparing to save {len(similarities_dict)} items to Firestore...")
    saved_count = 0
    error_count = 0
    batch = db.batch() # Use batch writes for efficiency
    batch_count = 0
    max_batch_size = 490 # Firestore batch limit is 500 operations

    for outfit_id, neighbors in similarities_dict.items():
        try:
            # Document ID must be a string
            doc_id = str(outfit_id)
            doc_ref = db.collection(collection_name).document(doc_id)

            # Serialize the neighbors list to JSON string (safer for potential size)
            # Alternatively, store directly as array: {'neighbors': neighbors}
            # but be mindful of the 1 MiB document size limit.
            serialized_neighbors = json.dumps(neighbors)
            data = {
                'outfit_id': outfit_id, # Optional: Store original ID also in field
                'neighbors_json': serialized_neighbors
                # Could add a timestamp field: 'last_updated': firestore.SERVER_TIMESTAMP
            }

            batch.set(doc_ref, data, merge=True) # merge=True acts as an upsert
            batch_count += 1
            saved_count += 1

            # Commit the batch periodically to avoid exceeding limits
            if batch_count >= max_batch_size:
                logging.info(f"Committing batch of {batch_count} items...")
                batch.commit()
                logging.info(f"Committed. Total saved so far: {saved_count}")
                # Start a new batch
                batch = db.batch()
                batch_count = 0

        except Exception as e:
            logging.error(f"Error preparing batch for outfit_id {outfit_id}: {e}")
            error_count += 1

    # Commit any remaining items in the last batch
    if batch_count > 0:
        try:
            logging.info(f"Committing final batch of {batch_count} items...")
            batch.commit()
            logging.info(f"Final batch committed.")
        except Exception as e:
            logging.error(f"Error committing final batch: {e}")
            error_count += batch_count # Approximate errors

    logging.info(f"Finished saving to Firestore. Successful: {saved_count}, Errors: {error_count}")
    if error_count > 0:
        logging.warning("Some items failed to save to Firestore.")

# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info("Starting daily outfit similarity calculation job (using Firestore)...")

    # Call the function
    result = load_and_preprocess_data_from_gcs(
        GCS_BUCKET_NAME, INTERACTIONS_FILE_GCS_PATH, MIN_INTERACTIONS_PER_OUTFIT
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
                      save_similarities_to_firestore(
                           top_k_similar_outfits, FIRESTORE_COLLECTION_NAME
                      )
                      logging.info("\nOffline similarity calculation and Firestore update complete.")
                 else:
                      logging.info("\nNo similarities generated to save.")
            else:
                 logging.error("Outfit mapping is missing or invalid after preprocessing. Cannot calculate similarities.")
            # --- End calculation block ---
    else:
        logging.error(f"Failed to load or preprocess data correctly. Function returned: {result}. Exiting job.")
