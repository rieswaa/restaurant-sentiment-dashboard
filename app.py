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

# Navigasi
menu = st.sidebar.radio("Navigasi", [
    "Beranda",
    "Distribusi Rating",
    "Rating & Sentimen",
    "Wordcloud",
    "Reviewer & Restoran",
    "Review Positif & Negatif",
    "Kesimpulan Rating",
    "Data Mentah"
])

# Terapkan filter
filtered_df = df[
    (df['Rating'] >= rating_filter[0]) &
    (df['Rating'] <= rating_filter[1]) &
    (df['Time'] >= pd.to_datetime(start_date)) &
    (df['Time'] <= pd.to_datetime(end_date))
]
if selected_restaurant != "Semua":
    filtered_df = filtered_df[filtered_df['Restaurant'] == selected_restaurant]

# SLIDE: BERANDA
if menu == "Beranda":
    st.markdown("### 🔢 Statistik Review")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Review", len(filtered_df))
    col2.metric("Rata-rata Rating", f"{filtered_df['Rating'].mean():.2f}")
    col3.metric("Review Positif", f"{(filtered_df['Sentiment'] == 'Positive').mean()*100:.1f}%")
    col4.metric("Review Negatif", f"{(filtered_df['Sentiment'] == 'Negative').mean()*100:.1f}%")

# SLIDE: Distribusi Rating
elif menu == "Distribusi Rating":
    st.markdown("### 📈 Distribusi Rating")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, x='Rating', palette='Set2', ax=ax1)
    ax1.set_title("Distribusi Rating", fontsize=14)
    st.pyplot(fig1)

# SLIDE: Rating dan Sentimen
elif menu == "Rating & Sentimen":
    st.markdown("### 📉 Rating berdasarkan Sentimen")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=filtered_df, x='Sentiment', y='Rating', palette='pastel', ax=ax2)
    ax2.set_title("Distribusi Rating berdasarkan Sentimen", fontsize=14)
    st.pyplot(fig2)

# SLIDE: Wordcloud
elif menu == "Wordcloud":
    st.markdown("### ☁️ WordCloud Review")
    text = " ".join(filtered_df['Review'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=1000, height=300, background_color='#1c1c1c', colormap='Set2').generate(text)
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

# SLIDE: Top Reviewer & Restoran
elif menu == "Reviewer & Restoran":
    st.markdown("### 🏅 Reviewer & Restoran Teratas")
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        st.markdown("**👥 Top 5 Reviewer**")
        st.dataframe(filtered_df['Reviewer'].value_counts().head(5))
    with col_top2:
        st.markdown("**🍽️ Restoran Paling Banyak Direview**")
        st.bar_chart(filtered_df['Restaurant'].value_counts().head(5))

# SLIDE: Review Positif & Negatif
elif menu == "Review Positif & Negatif":
    st.markdown("### 💬 Contoh Review Positif & Negatif")
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("**✅ Positif**")
        pos_reviews = filtered_df[filtered_df['Sentiment'] == 'Positive']['Review']
        if len(pos_reviews) > 0:
            st.write(pos_reviews.sample(3, random_state=1).tolist())
        else:
            st.write("⚠️ Tidak ada review positif.")

    with col_neg:
        st.markdown("**❌ Negatif**")
        neg_reviews = filtered_df[filtered_df['Sentiment'] == 'Negative']['Review']
        if len(neg_reviews) > 0:
            st.write(neg_reviews.sample(3, random_state=1).tolist())
        else:
            st.write("⚠️ Tidak ada review negatif.")

# SLIDE: Kesimpulan Rating
elif menu == "Kesimpulan Rating":
    st.markdown("### 📌 Kesimpulan Alasan Pengguna Memberi Rating")
    for rating in range(1, 6):
        subset = filtered_df[filtered_df['Rating'] == rating]
        if subset.empty:
            continue
        words = ' '.join(subset['Review'].dropna().astype(str).tolist()).lower()
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', words)
        stopwords = set(WordCloud().stopwords)
        filtered_words = [w for w in tokens if w not in stopwords]
        common = Counter(filtered_words).most_common(8)
        word_list = ', '.join([w for w, _ in common])
        sample = subset['Review'].sample(1).iloc[0] if not subset.empty else "Tidak ada contoh."
        
        st.markdown(f"""
        #### ⭐ Rating {rating}
        - **Kata dominan:** {word_list}
        - **Contoh komentar:** _\"{sample}\"_
        """)

# SLIDE: Data Mentah
elif menu == "Data Mentah":
    st.markdown("### 📄 Data Mentah")
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Data CSV", data=csv, file_name='filtered_reviews.csv', mime='text/csv')
