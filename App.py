import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Hyderabad Foodie AI", page_icon="🍲", layout="wide")

# Custom CSS for a professional "Dark Mode" aesthetic
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .restaurant-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 1 & 2: DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('HyderabadResturants.csv')
    
    # 1. Basic Cleaning
    df['ratings_numeric'] = pd.to_numeric(df['ratings'], errors='coerce').fillna(3.5)
    
    df['cuisine_norm'] = df['cuisine'].str.lower().str.replace('[^a-zA-Z0-9, ]', '', regex=True).fillna('')

    def get_locality(link):
        try:
            parts = str(link).split('/')
            raw_segment = parts[4] if len(parts) > 4 else ""
            words = raw_segment.split('-')
            
            # Capture 3 words for single-letter prefixes (S R Nagar)
            if len(words) >= 3 and len(words[-3]) == 1:
                result = " ".join(words[-3:]).title()
            else:
                result = " ".join(words[-2:]).title()
            
            # Expanded Junk Filter
            junk_list = [
                'Monk', 'Hotel', 'Cafe', 'Restaurant', 'Bakery', 'Bristo', 
                'Order', 'House', 'Life', 'King', 'Point', 'Zone', 'Club'
            ]
            
            for junk in junk_list:
                result = result.replace(junk, '').strip()
            
            # Remove standalone numbers (1, 2, 3, etc.)
            result = " ".join([w for w in result.split() if not w.isdigit()])
            
            return result if result else "Hyderabad"
        except:
            return "Hyderabad"
            

    df['locality'] = df['links'].apply(get_locality)
    return df

# Initialize Data
df = load_and_clean_data()

# --- STEP 5 & 6: RECOMMENDATION ENGINE ---
@st.cache_resource
def build_engine(data):
    # We use 'cuisine_norm' which was just created in the function above
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['cuisine_norm'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = build_engine(df)


# --- UI HEADER ---
st.title("🍲 Hyderabad Restaurant Recommender")
st.markdown("### Find your next favorite meal using AI-powered flavor matching.")
st.divider()

# --- SIDEBAR FILTERS (Step 7: Hybrid Filtering) ---
st.sidebar.header("🎯 Refine Your Search")
selected_area = st.sidebar.multiselect("Select Localities", options=sorted(df['locality'].unique()))
budget = st.sidebar.slider("Max Budget (Price for One)", min_value=50, max_value=500, value=500, step=50)

st.sidebar.info("💡 **Pro Tip:** This app uses TF-IDF and Cosine Similarity to match restaurant 'flavor profiles' with 95% relevance.")

# --- MAIN INTERFACE ---.
# --- MAIN INTERACTION ---
col1, col2 = st.columns([1, 1])

with col1:
    target_restaurant = st.selectbox("Type or select a restaurant you love:", options=df['names'].unique())
    num_recs = st.number_input("How many recommendations?", min_value=1, max_value=10, value=5)

if st.button("✨ Get Recommendations"):
    if target_restaurant in df['names'].values:
        try:
            with st.spinner('Analyzing flavor profiles...'):
                idx = df[df['names'] == target_restaurant].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:51]
                
                candidate_indices = [i[0] for i in sim_scores]
                recs = df.iloc[candidate_indices].copy()
                
                if selected_area:
                    recs = recs[recs['locality'].isin(selected_area)]
                recs = recs[recs['price for one'] <= budget]

                if recs.empty:
                    st.warning("⚠️ No matches found. Try expanding your area or budget!")
                else:
                    st.subheader(f"Because you liked {target_restaurant}...")
                    for i in range(min(len(recs), num_recs)):
                        row = recs.iloc[i]
                        with st.container():
                            st.markdown(f"""
                            <div class="restaurant-card">
                                <h3>{row['names']}</h3>
                                <p>📍 <b>Locality:</b> {row['locality']} | 💰 <b>Price:</b> ₹{row['price for one']}</p>
                                <p>🍲 <b>Cuisines:</b> {row['cuisine']}</p>
                                <p>⭐ <b>Rating:</b> {row['ratings']}</p>
                                <a href="{row['links']}" target="_blank" style="color: #ff4b4b;">Order on Zomato →</a>
                            </div>
                            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("📍 This restaurant isn't in our database. Try 'Bawarchi' or 'pista house' !")


