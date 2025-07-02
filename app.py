import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import re

# 🎨 Setup halaman
st.set_page_config(page_title="Restaurant Sentiment", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #6c5ce7;'>🍽️ Restaurant Sentiment Dashboard</h1>",
    unsafe_allow_html=True
)

# 📊 Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Restaurant reviews.csv")
    df = df.drop(columns=['7514'], errors='ignore')
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Review', 'Rating'])
    return df

df = load_data()

# 🧠 Analisis sentimen
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

# 🎛️ Sidebar filter
st.sidebar.header("🎯 Filter Data")
rating_filter = st.sidebar.slider("Rating", 1, 5, (1, 5))
restaurant_options = df['Restaurant'].dropna().unique().tolist()
selected_restaurant = st.sidebar.selectbox("Pilih Restoran", ["Semua"] + restaurant_options)
start_date = st.sidebar.date_input("Tanggal Mulai", df['Time'].min().date())
end_date = st.sidebar.date_input("Tanggal Akhir", df['Time'].max().date())

# 🔍 Terapkan filter
filtered_df = df[
    (df['Rating'] >= rating_filter[0]) &
    (df['Rating'] <= rating_filter[1]) &
    (df['Time'] >= pd.to_datetime(start_date)) &
    (df['Time'] <= pd.to_datetime(end_date))
]
if selected_restaurant != "Semua":
    filtered_df = filtered_df[filtered_df['Restaurant'] == selected_restaurant]

# 📊 METRIK
st.markdown("### 🔢 Statistik Review")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Review", len(filtered_df))
col2.metric("Rata-rata Rating", f"{filtered_df['Rating'].mean():.2f}")
col3.metric("Review Positif", f"{(filtered_df['Sentiment'] == 'Positive').mean()*100:.1f}%")
col4.metric("Review Negatif", f"{(filtered_df['Sentiment'] == 'Negative').mean()*100:.1f}%")

# 📈 Distribusi Rating
st.markdown("### 📈 Distribusi Rating")
fig1, ax1 = plt.subplots(figsize=(7, 4))
sns.countplot(data=filtered_df, x='Rating', palette='Set2', ax=ax1)
ax1.set_title("Distribusi Rating", fontsize=14, color='white')
ax1.set_xlabel("Rating", fontsize=12, color='white')
ax1.set_ylabel("Jumlah", fontsize=12, color='white')
ax1.tick_params(colors='white')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_facecolor('#0e1117')
fig1.patch.set_facecolor('#0e1117')
st.pyplot(fig1)

# 📉 Boxplot Rating vs Sentimen
st.markdown("### 📉 Rating berdasarkan Sentimen")
fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.boxplot(data=filtered_df, x='Sentiment', y='Rating', palette='pastel', ax=ax2)
ax2.set_title("Distribusi Rating berdasarkan Sentimen", fontsize=14, color='white')
ax2.set_xlabel("Sentimen", fontsize=12, color='white')
ax2.set_ylabel("Rating", fontsize=12, color='white')
ax2.tick_params(colors='white')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_facecolor('#0e1117')
fig2.patch.set_facecolor('#0e1117')
st.pyplot(fig2)

# ☁️ WordCloud
st.markdown("### ☁️ WordCloud Review")
text = " ".join(filtered_df['Review'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=1000, height=300, background_color='#1c1c1c', colormap='Set2').generate(text)
fig3, ax3 = plt.subplots(figsize=(10, 3.5))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis('off')
fig3.patch.set_facecolor('#0e1117')
st.pyplot(fig3)

# 🏅 Top Reviewer & Restoran
st.markdown("### 🏅 Reviewer & Restoran Teratas")
col_top1, col_top2 = st.columns(2)
with col_top1:
    st.markdown("**👥 Top 5 Reviewer**")
    st.dataframe(filtered_df['Reviewer'].value_counts().head(5))
with col_top2:
    st.markdown("**🍽️ Restoran Paling Banyak Direview**")
    st.bar_chart(filtered_df['Restaurant'].value_counts().head(5))

# 💬 Contoh Review Positif & Negatif
st.markdown("### 💬 Contoh Review Positif & Negatif")
col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("**✅ Positif**")
    pos_reviews = filtered_df[filtered_df['Sentiment'] == 'Positive']['Review']
    if len(pos_reviews) >= 3:
        st.write(pos_reviews.sample(3, random_state=1).tolist())
    elif len(pos_reviews) > 0:
        st.write(pos_reviews.tolist())
    else:
        st.write("⚠️ Tidak ada review positif untuk filter saat ini.")

with col_neg:
    st.markdown("**❌ Negatif**")
    neg_reviews = filtered_df[filtered_df['Sentiment'] == 'Negative']['Review']
    if len(neg_reviews) >= 3:
        st.write(neg_reviews.sample(3, random_state=1).tolist())
    elif len(neg_reviews) > 0:
        st.write(neg_reviews.tolist())
    else:
        st.write("⚠️ Tidak ada review negatif untuk filter saat ini.")

# 📌 Kesimpulan Alasan Rating
st.markdown("### 📌 Kesimpulan Alasan Pengguna Memberi Rating")
st.markdown("""
**✅ Positif:**
- Pengguna puas dengan menu seperti *paratha*, *ginger tea*, *apple pie*.
- Cita rasa dan kualitas makanan menjadi alasan utama memberikan rating tinggi.
- Kata seperti **loved**, **best**, dan **tastier** sering muncul.

**❌ Negatif:**
- Pengguna kecewa karena porsi kecil (*half cup tea*), pengalaman buruk, atau pelayanan tidak memuaskan.
- Alasan kuat lainnya adalah kata seperti **worst**, **looting**, dan **ignored**.
""")

# 📥 Download Data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Unduh Data CSV", data=csv, file_name='filtered_reviews.csv', mime='text/csv')

# 📄 Data mentah
with st.expander("📄 Tampilkan Data Mentah"):
    st.dataframe(filtered_df)
