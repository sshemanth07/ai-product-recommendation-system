import pandas as pd
from tqdm import tqdm

def build_features(clients, events, product_props, sample_size=8000):
    sample_clients = clients[:sample_size]
    
    buys = events["product_buy"].merge(product_props, on="sku", how="left")
    buys = buys[buys["client_id"].isin(sample_clients)]

    features = []
    for cid in tqdm(sample_clients, desc="Building features"):
        row = {"client_id": cid}
        cb = buys[buys["client_id"] == cid]

        for c in range(100):
            row[f"buy_cat_{c}"] = (cb["category"] == c).sum()

        row["total_buys"] = len(cb)
        row["unique_skus"] = cb["sku"].nunique()
        row["avg_price"] = cb["price"].mean() if not cb.empty else 0.0
        row["page_visits"] = len(events["page_visit"][events["page_visit"]["client_id"] == cid])
        row["searches"] = len(events["search_query"][events["search_query"]["client_id"] == cid])

        features.append(row)

    df = pd.DataFrame(features)
    df["target"] = (df["total_buys"] <= 1).astype(int)

    return df
