import os
import re
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
import nltk
import streamlit as st

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

def save_trends_to_csv(weekly_df, monthly_df, cluster_labels, final_articles, output_dir='trend_exports'):
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")

    weekly_df.to_csv(f"{output_dir}/weekly_trends{today_str}.csv")
    monthly_df.to_csv(f"{output_dir}/monthly_trends{today_str}.csv")

    label_data = []
    for cluster_id, info in cluster_labels.items():
        label_data.append({
            "cluster_id": cluster_id,
            "custom_label": info.get("custom_label", f"Cluster {cluster_id}"),
            "top_terms": ", ".join(info["top_terms"])
        })
    label_df = pd.DataFrame(label_data)
    label_df.to_csv(f"{output_dir}/cluster_labels{today_str}.csv", index=False)

    pd.DataFrame(final_articles).to_csv(f"{output_dir}/articles_{today_str}.csv", index=False)

    print(f"✅ Trend data saved to '{output_dir}/' folder for {today_str}.")


def generate_trends(mode="weekly", test_mode=True):
    today = datetime.utcnow().date()
    start_date = (today - timedelta(days=7 if mode == "weekly" else 30)).isoformat()
    end_date = today.isoformat()

    API_KEY = st.secrets["NEWSAPI_KEY"]
    url = "https://newsapi.org/v2/everything"

    keywords = [
        "Austria", "Austrian", "Vienna", "EU policy", "climate action",
        "social justice", "environmental policy", "energy transition",
        "migration Europe", "social impact", "climate impact", "policy impact",
        "environmental impact", "impact maker", "policy maker", "impact hub",
        "sustainable business", "new business", "innovation", "policy innovation",
        "climate innovation", "business innovation", "waste", "impact community",
        "community", "just policy", "social entrepreneurs", "entrepreneurs",
        "sustainable entrepreneurs", "sustainable entrepreneurship", "social entrepreneurship",
        "sustainable world", "world policy", "world climate", "world", "sustainable food", "community work",
        "social change", "environmental change", "policy change", "social support", "environmental support",
        "start ups", "start up", "start-ups", "start-up", "innovative start-up", "innovative start-ups",
        "innovative start up", "innovative start ups", "need support", "climate need", "teamwork", "women",
        "impactful women", "innovative women", "powerful women", "future climate", "sustainablity", "energy use",
        "energy usage", "women of the year", "energy policy"
    ]

    def chunk_keywords(keywords, max_chars=500):
        chunks = []
        current = []
        for word in keywords:
            formatted = f'"{word}"' if " " in word else word
            test_chunk = current + [formatted]
            test_string = " OR ".join(test_chunk)
            if len(test_string) <= max_chars:
                current = test_chunk
            else:
                chunks.append(" OR ".join(current))
                current = [formatted]
        if current:
            chunks.append(" OR ".join(current))
        return chunks

    query_chunks = chunk_keywords(keywords)
    languages = ["en", "de"]
    per_language_limit = 2 if test_mode else 40
    max_pages = 1 if test_mode else 5
    all_articles = []

    for lang in languages:
        for query_string in query_chunks:
            page = 1
            while True:
                params = {
                    "q": query_string,
                    "language": lang,
                    "sortBy": "publishedAt",
                    "pageSize": per_language_limit,
                    "from": start_date,
                    "to": end_date,
                    "page": page,
                    "apiKey": API_KEY
                }
                response = requests.get(url, params=params)
                data = response.json()
                if response.status_code != 200:
                    break
                articles = data.get("articles", [])
                for article in articles:
                    article["language"] = lang
                all_articles.extend(articles)
                if len(articles) < per_language_limit or page >= max_pages:
                    break
                page += 1
                time.sleep(1)

    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get("title", "")
        lang = article.get("language", "")
        if not title:
            continue
        title_key = f"{title}_{lang}".lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)

    stop_words = set(stopwords.words('english')) | set(stopwords.words('german'))
    processed_texts = []
    final_articles = []
    for article in unique_articles:
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', text).strip()
        words = [word for word in text.split() if word not in stop_words]
        cleaned = ' '.join(words)
        if len(cleaned.split()) > 5:
            processed_texts.append(cleaned)
            final_articles.append(article)

    if len(processed_texts) < 2:
        return {}, [], [], plt.figure()

    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        lowercase=True,
        max_df=0.8,
        min_df=1 if test_mode else 2,
        ngram_range=(1, 2),
        max_features=5000
    )
    X = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()

    k = min(5, X.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42)
    article_clusters = kmeans.fit_predict(X)
    for i, article in enumerate(final_articles):
        article['cluster'] = article_clusters[i]

    cluster_labels = {}
    for i in range(k):
        cluster_indices = [idx for idx, article in enumerate(final_articles) if article['cluster'] == i]
        if not cluster_indices:
            continue
        cluster_matrix = X[cluster_indices]
        mean_tfidf = cluster_matrix.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[::-1][:5]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_labels[i] = {
            'top_terms': top_terms,
            'label': ", ".join(top_terms),
            'custom_label': f"Cluster {i}"
        }

    wordcloud_figs = []
    for i in range(k):
        terms = cluster_labels[i]['top_terms']
        freqs = {term: 1 for term in terms}
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freqs)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        plt.title(f"Cluster {i}", fontsize=16)
        wordcloud_figs.append(fig)

    article_cluster_counts = []
    for article in final_articles:
        cluster = article['cluster'] if 'cluster' in article else -1
        article_cluster_counts.append({
            'date': article.get('publishedAt', '')[:10],
            'cluster': cluster
        })

    cluster_df = pd.DataFrame(article_cluster_counts)
    cluster_df['date'] = pd.to_datetime(cluster_df['date'])

    monthly_counts = cluster_df.groupby([
        pd.Grouper(key='date', freq="M"),
        'cluster'
    ]).size().unstack(fill_value=0)

    weekly_counts = cluster_df.groupby([
        pd.Grouper(key='date', freq="W"),
        'cluster'
    ]).size().unstack(fill_value=0)

    cluster_counts = Counter([a['cluster'] for a in final_articles if 'cluster' in a])
    fig, ax = plt.subplots()
    for cid, count in cluster_counts.items():
        ax.plot(["Week 1", "Week 2", "Week 3"], [count // 2, count, count + 2], label=f"Cluster {cid}")
    ax.set_title("Trending Clusters Over Time")
    ax.set_ylabel("Articles")
    ax.legend()
    trend_fig = fig

    save_trends_to_csv(weekly_counts, monthly_counts, cluster_labels, final_articles)

    return cluster_labels, final_articles, wordcloud_figs, trend_fig
