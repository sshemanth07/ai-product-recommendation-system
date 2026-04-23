from pathlib import Path
import numpy as np
import pandas as pd

# 🔧 ADD THESE
USE_SAMPLE = False        # change to False for full run
SAMPLE_SIZE = 10000      # adjust if needed

def load_data():
    INPUT_PATH = Path("data/input")
    PRODUCT_PATH = Path("data/product_properties.parquet")

    clients = np.load(INPUT_PATH / "relevant_clients.npy")
    
    events = {}
    for name in ["product_buy", "add_to_cart", "remove_from_cart", "page_visit", "search_query"]:
        path = INPUT_PATH / f"{name}.parquet"
        df = pd.read_parquet(path)

        # 🔥 ADD THIS BLOCK
        if USE_SAMPLE:
            df = df.head(min(SAMPLE_SIZE, len(df)))
        
        events[name] = df

    product_props = pd.read_parquet(PRODUCT_PATH)
    
    print(f"Loaded {len(clients):,} clients and {len(product_props):,} products")

    if USE_SAMPLE:
        print(f"Running in SAMPLE mode ({SAMPLE_SIZE} rows per event file)")

    return clients, events, product_props
