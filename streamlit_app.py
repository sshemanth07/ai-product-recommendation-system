import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Purchase Prediction App")

total_buys = st.number_input("Total Buys", 0)
unique_skus = st.number_input("Unique SKUs", 0)
avg_price = st.number_input("Avg Price", 0.0)
page_visits = st.number_input("Page Visits", 0)
searches = st.number_input("Searches", 0)

if st.button("Predict"):
    payload = {
        "total_buys": total_buys,
        "unique_skus": unique_skus,
        "avg_price": avg_price,
        "page_visits": page_visits,
        "searches": searches
    }

    res = requests.post(f"{API_URL}/predict", json=payload)

    if res.status_code == 200:
        st.success(res.json())
    else:
        st.error("API Error")
