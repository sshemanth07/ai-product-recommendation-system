import pandas as pd
from tqdm import tqdm


def build_features(clients, events, product_props, sample_size=8000):
    sample_clients = clients[:sample_size]

    features = []
    for cid in tqdm(sample_clients, desc="Building features"):
        row = {"client_id": cid}

        # Non-leaky behavioural features
        row["page_visits"] = len(
            events["page_visit"][events["page_visit"]["client_id"] == cid]
        )
        row["searches"] = len(
            events["search_query"][events["search_query"]["client_id"] == cid]
        )
        row["add_to_cart_count"] = len(
            events["add_to_cart"][events["add_to_cart"]["client_id"] == cid]
        )
        row["remove_from_cart_count"] = len(
            events["remove_from_cart"][events["remove_from_cart"]["client_id"] == cid]
        )

        # Target: whether the user made at least one purchase
        row["target"] = int(
            len(events["product_buy"][events["product_buy"]["client_id"] == cid]) > 0
        )

        features.append(row)

    df = pd.DataFrame(features)
    return df
