import pandas as pd
import numpy as np

def create_training_pairs(events, product_props, sample_clients, negatives_per_positive=2):
    buys = events["product_buy"]
    buys = buys[buys["client_id"].isin(sample_clients)]
    
    popular_skus = buys["sku"].value_counts().head(300).index.tolist()
    
    positive_pairs = buys[["client_id", "sku"]].drop_duplicates().copy()
    positive_pairs["label"] = 1
    
    bought_by_client = positive_pairs.groupby("client_id")["sku"].apply(set).to_dict()
    
    negative_rows = []
    rng = np.random.default_rng(42)
    
    for client, bought_items in bought_by_client.items():
        candidates = [sku for sku in popular_skus if sku not in bought_items]
        if len(candidates) == 0:
            continue
        
        n_neg = min(len(bought_items) * negatives_per_positive, len(candidates))
        sampled = rng.choice(candidates, size=n_neg, replace=False)
        
        for sku in sampled:
            negative_rows.append((client, sku, 0))
    
    negative_pairs = pd.DataFrame(negative_rows, columns=["client_id", "sku", "label"])
    
    training_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
    training_pairs = training_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Training pairs: {len(training_pairs)} (Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)})")
    
    return training_pairs, popular_skus

def build_product_features(product_props):
    product_features = product_props.copy()
    
    for col in product_features.columns:
        if product_features[col].dtype == "object":
            product_features[col] = product_features[col].astype("category").cat.codes
    
    product_features = product_features.fillna(0)
    product_features = product_features.set_index("sku")
    
    return product_features

def merge_features(training_pairs, user_features, product_features):
    df = training_pairs.merge(user_features, on="client_id", how="left")
    df = df.merge(product_features.reset_index(), on="sku", how="left")
    df = df.fillna(0)
    
    return df