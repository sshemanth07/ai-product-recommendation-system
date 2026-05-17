import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_features(clients, events, product_props, sample_size=150000):
    np.random.seed(42)
    if sample_size > len(clients):
        sample_size = len(clients)
    sample_clients = np.random.choice(clients, size=sample_size, replace=False)
    
    buys = events["product_buy"]
    buys = buys[buys["client_id"].isin(sample_clients)]
    
    buys_with_props = buys.merge(product_props, on="sku", how="left")
    
    purchase_count = buys.groupby("client_id").size()
    unique_products = buys.groupby("client_id")["sku"].nunique()
    
    df = pd.DataFrame({"client_id": sample_clients})
    df["purchase_count"] = df["client_id"].map(purchase_count).fillna(0)
    df["unique_products"] = df["client_id"].map(unique_products).fillna(0)
    
    if "price" in buys_with_props.columns:
        price_stats = buys_with_props.groupby("client_id")["price"].agg([
            "mean", "std", "min", "max", "median"
        ]).round(2)
        price_stats.columns = ["price_mean", "price_std", "price_min", "price_max", "price_median"]
        
        df["price_mean"] = df["client_id"].map(price_stats["price_mean"]).fillna(0)
        df["price_std"] = df["client_id"].map(price_stats["price_std"]).fillna(0)
        df["price_min"] = df["client_id"].map(price_stats["price_min"]).fillna(0)
        df["price_max"] = df["client_id"].map(price_stats["price_max"]).fillna(0)
        df["price_median"] = df["client_id"].map(price_stats["price_median"]).fillna(0)
        df["price_range"] = df["price_max"] - df["price_min"]
    
    if "category" in buys_with_props.columns:
        unique_categories = buys_with_props.groupby("client_id")["category"].nunique()
        df["unique_categories"] = df["client_id"].map(unique_categories).fillna(0)
        
        def get_top_category(group):
            return group.value_counts().index[0] if len(group) > 0 else -1
        
        top_category = buys_with_props.groupby("client_id")["category"].apply(get_top_category)
        df["top_category"] = df["client_id"].map(top_category).fillna(-1)
        
        category_counts = buys_with_props.groupby(["client_id", "category"]).size().unstack(fill_value=0)
        top_5_cats = category_counts.sum().nlargest(5).index.tolist()
        for cat in top_5_cats:
            df[f"category_{cat}_count"] = df["client_id"].map(category_counts[cat]).fillna(0)
    
    for event_name in ["page_visit", "search_query", "add_to_cart", "remove_from_cart"]:
        event_counts = events[event_name].groupby("client_id").size()
        df[f"{event_name}_count"] = df["client_id"].map(event_counts).fillna(0)
    
    if "sku" in events["page_visit"].columns:
        visit_unique = events["page_visit"].groupby("client_id")["sku"].nunique()
        df["unique_products_visited"] = df["client_id"].map(visit_unique).fillna(0)
    
    if "sku" in events["add_to_cart"].columns:
        cart_unique = events["add_to_cart"].groupby("client_id")["sku"].nunique()
        df["unique_products_carted"] = df["client_id"].map(cart_unique).fillna(0)
    
    df["cart_to_purchase_ratio"] = df["add_to_cart_count"] / (df["purchase_count"] + 1)
    df["visit_to_purchase_ratio"] = df["page_visit_count"] / (df["purchase_count"] + 1)
    df["search_to_purchase_ratio"] = df["search_query_count"] / (df["purchase_count"] + 1)
    df["cart_abandonment_rate"] = df["remove_from_cart_count"] / (df["add_to_cart_count"] + 1)
    
    df["engagement_score"] = (
        df["add_to_cart_count"] * 3 +
        df["page_visit_count"] * 1 +
        df["search_query_count"] * 2 -
        df["remove_from_cart_count"] * 2
    )
    
    df["product_diversity"] = (
        df["unique_products"] + 
        df.get("unique_categories", 0) + 
        df.get("unique_products_visited", 0)
    ) / 3
    
    df["target"] = (df["purchase_count"] >= 2).astype(int)
    df = df.drop(["purchase_count"], axis=1)
    
    leakage_cols = ["unique_products"]
    for col in leakage_cols:
        if col in df.columns:
            df = df.drop([col], axis=1)
    
    feature_cols = [col for col in df.columns if col not in ["client_id", "target"]]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"Built features: {len(df)} clients, {len(feature_cols)} features")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df