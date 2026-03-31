import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import folium
import os

# 1. Veri Yükleme
# Dosyanın python dosyasıyla aynı klasörde olduğunu varsayıyoruz
file_path = 'traffic_density_202501.csv'

if not os.path.exists(file_path):
    print(f"Hata: {file_path} bulunamadı! Lütfen veri setini klasöre ekleyin.")
else:
    df = pd.read_csv(file_path, nrows=100000)

    # 2. Ön İşleme
    df['datetime'] = pd.to_datetime(df['DATE_TIME'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    df = df.dropna()

    features = [
        'LATITUDE', 'LONGITUDE',
        'AVERAGE_SPEED', 'MINIMUM_SPEED', 'MAXIMUM_SPEED',
        'NUMBER_OF_VEHICLES',
        'hour', 'dayofweek', 'is_weekend'
    ]
    X = df[features]

    # 3. Ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. KMeans ve Kümeleme
    k_optimal = 3 
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 5. PCA ve Görselleştirme
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    # Plotly grafiğini tarayıcıda açar
    fig = px.scatter(
        df, x='PCA1', y='PCA2',
        color=df['cluster'].astype(str),
        title="PCA ile Trafik Küme Görselleştirme"
    )
    fig.show()

    # 6. Haritalama (Folium)
    # İstanbul merkezli harita
    map_clusters = folium.Map(location=[41.015137, 28.979530], zoom_start=10)
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Önemli: Çok fazla nokta (100k) haritayı kilitleyebilir, görsel için ilk 1000 örneği basalım
    for i, row in df.head(1000).iterrows():
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=3,
            color=colors[int(row['cluster']) % len(colors)],
            fill=True,
            fill_opacity=0.6
        ).add_to(map_clusters)

    # Haritayı HTML olarak kaydet (VS Code'da doğrudan görünmez, bu dosyayı tarayıcıda açmalısınız)
    map_clusters.save("istanbul_trafik_haritasi.html")
    print("Analiz tamamlandı. Harita 'istanbul_trafik_haritasi.html' olarak kaydedildi.")