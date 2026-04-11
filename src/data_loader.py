from pathlib import Path
import numpy as np
import pandas as pd

def load_data():
    INPUT_PATH = Path("data/raw/input")
    PRODUCT_PATH = Path("data/raw/product_properties.parquet")

    clients = np.load(INPUT_PATH / "relevant_clients.npy")
    
    events = {}
    for name in ["product_buy", "add_to_cart", "remove_from_cart", "page_visit", "search_query"]:
        path = INPUT_PATH / f"{name}.parquet"
        events[name] = pd.read_parquet(path)

    product_props = pd.read_parquet(PRODUCT_PATH)
    
    print(f"Loaded {len(clients):,} clients and {len(product_props):,} products")
    return clients, events, product_props
