import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import random
from collections import Counter
import re

# Setup halaman
st.set_page_config(page_title="Restaurant Sentiment", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #6c5ce7;'>🍽️ Restaurant Sentiment Dashboard</h1>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Restaurant reviews.csv")
    df = df.drop(columns=['7514'], errors='ignore')
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Review', 'Rating'])
    return df

df = load_data()

# Analisis sentimen
@st.cache_data
def analyze_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Review'].apply(analyze_sentiment)

# Sidebar filter
st.sidebar.header("🎯 Filter Data")
rating_filter = st.sidebar.slider("Rating", 1, 5, (1, 5))
restaurant_options = df['Restaurant'].dropna().unique().tolist()
selected_restaurant = st.sidebar.selectbox("Pilih Restoran", ["Semua"] + restaurant_options)
start_date = st.sidebar.date_input("Tanggal Mulai", df['Time'].min().date())
end_date = st.sidebar.date_input("Tanggal Akhir", df['Time'].max().date())

# Terapkan filter
filtered_df = df[
    (df['Rating'] >= rating_filter[0]) &
    (df['Rating'] <= rating_filter[1]) &
    (df['Time'] >= pd.to_datetime(start_date)) &
    (df['Time'] <= pd.to_datetime(end_date))
]
if selected_restaurant != "Semua":
    filtered_df = filtered_df[filtered_df['Restaurant'] == selected_restaurant]

# Tabs Navigasi
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Statistik", 
    "📈 Distribusi Rating", 
    "📉 Rating & Sentimen", 
    "☁️ WordCloud", 
    "🏅 Reviewer & Restoran", 
    "💬 Positif & Negatif", 
    "🧠 Kesimpulan Rating", 
    "📄 Data Mentah"
])

# Tab 1 - Statistik
with tab1:
    st.markdown("### 🔢 Statistik Review")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Review", len(filtered_df))
    col2.metric("Rata-rata Rating", f"{filtered_df['Rating'].mean():.2f}")
    col3.metric("Review Positif", f"{(filtered_df['Sentiment'] == 'Positive').mean()*100:.1f}%")
    col4.metric("Review Negatif", f"{(filtered_df['Sentiment'] == 'Negative').mean()*100:.1f}%")

# Tab 2 - Distribusi Rating
with tab2:
    st.markdown("### 📈 Distribusi Rating")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, x='Rating', palette='Set2', ax=ax1)
    ax1.set_title("Distribusi Rating", fontsize=14)
    st.pyplot(fig1)

# Tab 3 - Rating dan Sentimen
with tab3:
    st.markdown("### 📉 Rating berdasarkan Sentimen")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=filtered_df, x='Sentiment', y='Rating', palette='pastel', ax=ax2)
    ax2.set_title("Distribusi Rating berdasarkan Sentimen", fontsize=14)
    st.pyplot(fig2)

# Tab 4 - Wordcloud
with tab4:
    st.markdown("### ☁️ WordCloud Review")
    text = " ".join(filtered_df['Review'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=1000, height=300, background_color='#1c1c1c', colormap='Set2').generate(text)
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

# Tab 5 - Reviewer & Restoran
with tab5:
    st.markdown("### 🏅 Reviewer & Restoran Teratas")
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        st.markdown("**👥 Top 5 Reviewer**")
        st.dataframe(filtered_df['Reviewer'].value_counts().head(5))
    with col_top2:
        st.markdown("**🍽️ Restoran Paling Banyak Direview**")
        st.bar_chart(filtered_df['Restaurant'].value_counts().head(5))

# Tab 6 - Review Positif & Negatif
with tab6:
    st.markdown("### 💬 Contoh Review Positif & Negatif")
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("**✅ Positif**")
        pos_reviews = filtered_df[filtered_df['Sentiment'] == 'Positive']['Review']
        if len(pos_reviews) >= 3:
            st.write(pos_reviews.sample(3).tolist())
        elif len(pos_reviews) > 0:
            st.write(pos_reviews.tolist())
        else:
            st.write("⚠️ Tidak ada review positif.")

    with col_neg:
        st.markdown("**❌ Negatif**")
        neg_reviews = filtered_df[filtered_df['Sentiment'] == 'Negative']['Review']
        if len(neg_reviews) >= 3:
            st.write(neg_reviews.sample(3).tolist())
        elif len(neg_reviews) > 0:
            st.write(neg_reviews.tolist())
        else:
            st.write("⚠️ Tidak ada review negatif.")

# Tab 7 - Kesimpulan Rating
st.markdown("### 📌 Kesimpulan Alasan Pengguna Memberi Rating")

rating_summary = {
    1: {"kata": "veg, chicken, food, burger, pizza, bad, experience, like", "kesimpulan": "Pengguna kecewa..."},
    2: {"kata": "good, food, place, chicken, service, taste, ambience, price", "kesimpulan": "Walau ada..."},
    3: {"kata": "good, taste, food, place, service, chicken, biryani, try", "kesimpulan": "Review netral..."},
    4: {"kata": "good, cake, food, chicken, best, try, chocolate", "kesimpulan": "Rating tinggi..."},
    5: {"kata": "good, food, place, service, great, best, chicken, ambience, really", "kesimpulan": "Pengguna sangat puas..."}
}

for rating in range(rating_filter[0], rating_filter[1] + 1):
    subset = filtered_df[filtered_df['Rating'] == rating]
    if len(subset) == 0:
        continue
    st.markdown(f"#### ⭐ Rating {rating}")
    st.markdown(f"- **Kata dominan:** {rating_summary[rating]['kata']}")
    st.markdown(f"- **Kesimpulan:** {rating_summary[rating]['kesimpulan']}")

    if st.button(f"🔁 Acak komentar rating {rating}", key=f"btn_{rating}"):
        example = subset['Review'].dropna().sample(1).values[0]
        st.markdown(f"- **📝 Contoh Komentar:** _\"{example}\"_")
    else:
        example = subset['Review'].dropna().sample(1, random_state=rating).values[0]
        st.markdown(f"- **📝 Contoh Komentar:** _\"{example}\"_")

# Tab 8 - Data Mentah
with tab8:
    st.markdown("### 📄 Data Mentah")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Data CSV", data=csv, file_name='filtered_reviews.csv', mime='text/csv')
