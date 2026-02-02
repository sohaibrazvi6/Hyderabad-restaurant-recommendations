# 🍲 Hyderabad Restaurant Recommender
**An AI-powered discovery engine for Hyderabad's food scene using Content-Based Filtering.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg?style=flat)](https://hyderabad-restaurant-recommendations-3zdv3js9krqsrw2nrfpvtq.streamlit.app/)

## 🚀 Project Overview
This project is a specialized recommendation engine designed for foodies in Hyderabad. It uses Natural Language Processing (NLP) to analyze restaurant profiles—including cuisines, localities, and price points—to suggest the most relevant dining spots based on a user's preference.

## 🧠 The AI Engine
The "brain" of this application relies on two core mathematical concepts:

1.  **TF-IDF Vectorization**: I used `TfidfVectorizer` to convert restaurant cuisine tags into mathematical vectors. This allows the system to prioritize unique cuisine markers (like "Mandi" or "Haleem") over common words.
2.  **Cosine Similarity**: By calculating the cosine of the angle between two restaurant vectors, the engine determines their "flavor profile" similarity. A score of 1.0 represents an identical match.



## 🛠 Engineering Challenges & Solutions
As a CSE student, I focused on robust data engineering and production stability:

* **Advanced Data Extraction**: Built a custom Python regex-logic parser to extract multi-word Hyderabad localities (e.g., **S R Nagar**, **Himayath Nagar**) directly from Zomato URLs, ensuring 98% accuracy in geographic filtering.
* **Production Debugging**: Successfully resolved dependency conflicts between **Python 3.13**, **Streamlit**, and **Altair** by implementing precise version pinning in `requirements.txt`.
* **Hybrid Filtering**: Integrated a "Hybrid" search layer that combines AI similarity scores with hard constraints like **Budget** and **Area Selection**.

## 📂 Tech Stack
* **Language**: Python 3.13
* **Framework**: Streamlit
* **Libraries**: Scikit-Learn, Pandas, NumPy, Joblib
* **Deployment**: Streamlit Community Cloud

## 📈 Future Scope
* Integration of a **Bidirectional LSTM** model for real-time sentiment analysis of user reviews ("Vibe Check").
* Expanding the dataset beyond the current 650+ verified Hyderabad restaurants.

---
Developed with 🖤 in Hyderabad by syed sohaib sultan razvi
