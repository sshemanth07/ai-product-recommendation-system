import pandas as pd
import numpy as np

def define_target_from_validation(df_features, validation_target_path="data/raw/target/validation_target.parquet"):
    """
    Define target based on validation period data
    """
    try:
        val_df = pd.read_parquet(validation_target_path)
        
        print(f"Validation columns: {val_df.columns.tolist()}")
        
        # Try to find the right column names
        client_col = 'client_id' if 'client_id' in val_df.columns else val_df.columns[0]
        
        # Check if there's an event type column
        if 'event_type' in val_df.columns:
            val_buys = val_df[val_df['event_type'] == 'product_buy']
        else:
            # Assume all rows are interactions, count them
            val_buys = val_df
        
        # Count interactions per client
        if 'sku' in val_buys.columns or 'product_id' in val_buys.columns:
            sku_col = 'sku' if 'sku' in val_buys.columns else 'product_id'
            val_counts = val_buys.groupby(client_col)[sku_col].nunique().reset_index(name='future_purchases')
        else:
            val_counts = val_buys.groupby(client_col).size().reset_index(name='future_purchases')
        
        # Merge with features
        df_with_target = df_features.merge(val_counts, left_on='client_id', right_on=client_col, how='left')
        df_with_target['future_purchases'] = df_with_target['future_purchases'].fillna(0)
        
        # Target: client made any purchase in validation period
        df_with_target['target'] = (df_with_target['future_purchases'] >= 1).astype(int)
        
        print(f"Target distribution: {df_with_target['target'].value_counts().to_dict()}")
        
        return df_with_target
        
    except Exception as e:
        print(f"Validation error: {e}")
        print("Falling back to synthetic target")
        return create_synthetic_target_for_testing(df_features)

def create_synthetic_target_for_testing(df_features):
    """Fallback target based on engagement"""
    engagement = (
        df_features.get('page_visits', 0) * 0.3 +
        df_features.get('searches', 0) * 0.3 +
        df_features.get('cart_adds', 0) * 0.4
    )
    
    if engagement.std() > 0:
        engagement = (engagement - engagement.mean()) / engagement.std()
    
    prob = 1 / (1 + np.exp(-engagement))
    np.random.seed(42)
    df_features['target'] = (np.random.random(len(df_features)) < prob).astype(int)
    
    print(f"Synthetic target: {df_features['target'].value_counts().to_dict()}")
    
    return df_features