import umap
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import random
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from annotated_text import annotated_text
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def load_and_extract_audio_features(mp3_file):
    try:
        # Load audio file
        y, sr = librosa.load(mp3_file, sr=16000)
        features = {}

        # 1. Chroma STFT (Short-Time Fourier Transform)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)

        # 2. Root Mean Square (RMS) Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)

        # 3. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_var'] = np.var(spectral_centroid)

        # 4. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)

        # 6. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features['flatness_mean'] = np.mean(flatness)
        features['flatness_var'] = np.var(flatness)

        # 7. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_var'] = np.var(contrast)

        # 8. Spectral Flux
        flux = librosa.onset.onset_strength(y=y, sr=sr)
        features['flux_mean'] = np.mean(flux)
        features['flux_var'] = np.var(flux)

        # 9. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_var'] = np.var(zcr)

        # 10. Harmonics and Percussives
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmonics_mean'] = np.mean(y_harmonic)
        features['harmonics_var'] = np.var(y_harmonic)
        features['percussive_mean'] = np.mean(y_percussive)
        features['percussive_var'] = np.var(y_percussive)

        # 11. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        # 12. MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_var'] = np.var(mfccs[i])

        return y, sr, features
    except Exception as e:
        st.write(f"Error processing audio: {e}")
        return None, None, None

def make_predictions(audio_features, model):
    if model == "KMeans":
        model = joblib.load("models/kmeans_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        dim_reduction_model = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.0, random_state=42)
    elif model == "Agglomerative":
        model = joblib.load("models/agglomerative_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        dim_reduction_model = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.0, random_state=42)
    elif model == "GMM":
        model = joblib.load("models/gmm_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        dim_reduction_model = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.0, random_state=42)
    elif model == "Mean Shift":
        model = joblib.load("models/meanshift_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        dim_reduction_model = TSNE(n_components=2, perplexity=50, learning_rate=1000, n_iter=1000)
    elif model == "Birch":
        model = joblib.load("models/birch_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        dim_reduction_model = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.0, random_state=42)
    elif model == "DBSCAN":
        model = joblib.load("models/dbscan_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        dim_reduction_model = joblib.load("models/pca_model.pkl")
    
    
    df = pd.read_csv("dataset_30s.csv")
    df =df.drop_duplicates(keep='first')
    scaled_df = scaler.transform(df)

    audio_features_df = pd.DataFrame(audio_features, index=[0])
    
    combined_df = pd.concat([df, audio_features_df], axis=0)
    combined_df = scaler.transform(combined_df)
    ori_umap = dim_reduction_model.fit_transform(combined_df)

    # Transform the audio features
    audio_features_df_umap = ori_umap[-1].reshape(1, -1)
    audio_features_df_umap = audio_features_df_umap.astype(np.float32)
    combined_ori_umap = np.vstack((ori_umap, audio_features_df_umap))
    # Fit Model on the UMAP-reduced data
    ori_model_labels = model.fit_predict(combined_ori_umap)
    prediction = ori_model_labels[-1]
    ori_model_labels = ori_model_labels[:-1]

    return prediction, ori_model_labels, ori_umap, audio_features_df_umap


@st.cache_data
def plot_spectral_centroid(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroid))
    t = librosa.frames_to_time(frames, sr=sr)
    fig_centroid = go.Figure()
    fig_centroid.add_trace(go.Scatter(x=t, y=spectral_centroid, mode='lines', name='Spectral Centroid', line=dict(color='green')))
    fig_centroid.update_layout(title='Spectral Centroid', xaxis_title='Time (s)', yaxis_title='Hz', height=400)
    st.plotly_chart(fig_centroid, use_container_width=True)

@st.cache_data
def plot_spectral_rolloff(y, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    frames = range(len(spectral_rolloff))
    t = librosa.frames_to_time(frames, sr=sr)
    fig_rolloff = go.Figure()
    fig_rolloff.add_trace(go.Scatter(x=t, y=spectral_rolloff, mode='lines', name='Spectral Rolloff', line=dict(color='purple')))
    fig_rolloff.update_layout(title='Spectral Rolloff', xaxis_title='Time (s)', yaxis_title='Hz', height=400)
    st.plotly_chart(fig_rolloff, use_container_width=True)

@st.cache_data
def plot_stft(y, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig_stft = go.Figure(data=go.Heatmap(z=D, x=librosa.frames_to_time(np.arange(D.shape[1])), y=librosa.fft_frequencies(sr=sr), colorscale='Rainbow'))
    fig_stft.update_layout(title='Short-Time Fourier Transform (STFT)', xaxis_title='Time (s)', yaxis_title='Hz', height=400)
    st.plotly_chart(fig_stft, use_container_width=True)


def show_genre(prediction, model):
    if model == "Agglomerative":
        genres = [["hip-hop", "pop", "reggae"],
                  ["classical"],
                  ["metal", "rock"],
                  ["blues", "country", "disco", "jazz"]]
    elif model == "Mean Shift":
        genres = [["blues", "country", "disco", "metal", "rock"],
                  ["hip-hop", "pop", "reggae"],
                  ["classical", "jazz"]]
    elif model == "KMeans":
        genres = [["hip-hop", "pop", "reggae"],
                  ["classical"],
                  ["metal", "rock"],
                  ["blues", "country", "disco", "jazz"]]
    elif model == "GMM":
        genres = [["hip-hop", "jazz", "pop", "reggae"],
                  ["blues", "country", "disco", "metal", "rock"],
                  ["classical"],
                  ["unidentified"]]
    elif model == "Birch":
        genres = [["classical"],
                  ["unidentified"],
                  ["unidentified"],
                  ["unidentified"]]
    elif model == "DBSCAN":
        genres = [["unidentified"]]
    
    # Get the list of genres for the predicted cluster
    predicted_genres = genres[prediction]
    
    # Format the genres for the title
    formatted_genres = ", ".join(predicted_genres)
    
    # Display the title with all possible genres
    st.title(f"The genre may be {formatted_genres}")
    
    # Randomly select a genre from the predicted genres
    selected_genre = random.choice(predicted_genres)
    
    # Display the image for the selected genre
    try:
        image = st.image(f"genre_images/{selected_genre}.jpg", use_column_width=True)
    except FileNotFoundError:
        st.write(f"Image for {selected_genre} not found.")

    if model == "Agglomerative":
        if prediction == 0:
            annotated_text(
                ("Hip Hop, Pop, and Reggae", "genre", "#2d3e50"),
                " are genres that emerged in the ",
                ("mid to late 20th century", "time", "#e67e22"),
                ". Hip Hop originated in ",
                ("New York City", "origin", "#16a085"),
                ", Pop evolved globally, and Reggae came from ",
                ("Jamaica", "origin", "#16a085"),
                ". These genres share ",
                ("rhythmic focus", "characteristic", "#c0392b"),
                ", ",
                ("catchy melodies", "characteristic", "#c0392b"),
                ", and often address ",
                ("social issues", "theme", "#c0392b"),
                " and ",
                ("personal experiences", "theme", "#c0392b"),
                ". Common instruments include ",
                ("synthesizers", "instrument", "#2980b9"),
                ", ",
                ("drum machines", "instrument", "#2980b9"),
                ", and ",
                ("bass guitar", "instrument", "#2980b9"),
                ". All three have had significant ",
                ("cultural impact", "characteristic", "#c0392b"),
                " and ",
                ("global reach", "characteristic", "#c0392b"),
                ". Notable artists include ",
                ("Tupac Shakur", "artist", "#8e44ad"),
                ", ",
                ("Michael Jackson", "artist", "#8e44ad"),
                ", and ",
                ("Bob Marley", "artist", "#8e44ad"),
                "."
            )
        elif prediction == 1:
            annotated_text(
                ("Classical", "genre", "#2d3e50"),
                " music spans from the ",
                ("9th century", "time", "#e67e22"),
                " to the present, with significant periods including the ",
                ("Baroque", "period", "#d35400"),
                ", ",
                ("Classical", "period", "#d35400"),
                ", ",
                ("Romantic", "period", "#d35400"),
                ", and ",
                ("Contemporary", "period", "#d35400"),
                ". Rooted in Western liturgical and secular music traditions, it emphasizes ",
                ("formality", "characteristic", "#c0392b"),
                ", complexity, and orchestration, with structured compositions often featuring multiple movements. Key instruments include ",
                ("strings (violin, cello)", "instrument", "#2980b9"),
                ", ",
                ("woodwinds (flute, clarinet)", "instrument", "#2980b9"),
                ", ",
                ("brass (trumpet, trombone)", "instrument", "#2980b9"),
                ", and ",
                ("percussion (timpani)", "instrument", "#2980b9"),
                ". Renowned composers include ",
                ("Johann Sebastian Bach", "artist", "#8e44ad"),
                ", ",
                ("Ludwig van Beethoven", "artist", "#8e44ad"),
                ", and ",
                ("Wolfgang Amadeus Mozart", "artist", "#8e44ad"),
                " whose works continue to resonate in today's musical landscape."
            )
        elif prediction == 2:
            annotated_text(
                ("Metal and Rock", "genre", "#2d3e50"),
                " are closely related genres that emerged in the ",
                ("1950s to 1970s", "time", "#e67e22"),
                ". Both evolved from ",
                ("blues", "influence", "#d35400"),
                " and ",
                ("rhythm and blues", "influence", "#d35400"),
                ". These genres are characterized by their ",
                ("heavy use of electric guitars", "characteristic", "#c0392b"),
                ", ",
                ("strong rhythms", "characteristic", "#c0392b"),
                ", and ",
                ("energetic performances", "characteristic", "#c0392b"),
                ". Metal often features more ",
                ("aggressive sounds", "characteristic", "#c0392b"),
                " and ",
                ("complex compositions", "characteristic", "#c0392b"),
                ". Common instruments include ",
                ("electric guitar", "instrument", "#2980b9"),
                ", ",
                ("bass guitar", "instrument", "#2980b9"),
                ", and ",
                ("drums", "instrument", "#2980b9"),
                ". Influential artists span from rock icons like ",
                ("The Beatles", "artist", "#8e44ad"),
                " and ",
                ("Led Zeppelin", "artist", "#8e44ad"),
                " to metal pioneers such as ",
                ("Black Sabbath", "artist", "#8e44ad"),
                " and ",
                ("Metallica", "artist", "#8e44ad"),
                "."
            )
        elif prediction == 3:
            annotated_text(
                ("Blues, Country, Disco, and Jazz", "genre", "#2d3e50"),
                " are genres that developed primarily in the ",
                ("United States", "origin", "#16a085"),
                " from the ",
                ("late 19th to mid-20th century", "time", "#e67e22"),
                ". They share roots in ",
                ("African American musical traditions", "influence", "#d35400"),
                " and often feature ",
                ("emotional depth", "characteristic", "#c0392b"),
                " and ",
                ("storytelling", "characteristic", "#c0392b"),
                ". Blues and Country often address themes of ",
                ("love", "theme", "#c0392b"),
                ", ",
                ("loss", "theme", "#c0392b"),
                ", and ",
                ("hardship", "theme", "#c0392b"),
                ". Jazz and Disco emphasize ",
                ("complex rhythms", "characteristic", "#c0392b"),
                " and are closely tied to ",
                ("dance culture", "characteristic", "#c0392b"),
                ". Common instruments across these genres include ",
                ("guitar", "instrument", "#2980b9"),
                ", ",
                ("piano", "instrument", "#2980b9"),
                ", and various ",
                ("horns", "instrument", "#2980b9"),
                ". Notable artists include ",
                ("B.B. King", "artist", "#8e44ad"),
                " (Blues), ",
                ("Johnny Cash", "artist", "#8e44ad"),
                " (Country), ",
                ("Donna Summer", "artist", "#8e44ad"),
                " (Disco), and ",
                ("Miles Davis", "artist", "#8e44ad"),
                " (Jazz)."
            )
    elif model == "Mean Shift":
        if prediction == 0:
            annotated_text(
                ("Blues, Country, Disco, Metal, and Rock", "genre", "#2d3e50"),
                " are diverse genres that developed primarily in the ",
                ("United States", "origin", "#16a085"),
                " from the ",
                ("late 19th to mid-20th century", "time", "#e67e22"),
                ". Despite their differences, these genres share some common elements. They all have roots in ",
                ("earlier American musical traditions", "influence", "#d35400"),
                ", particularly ",
                ("blues", "influence", "#d35400"),
                ". These genres often feature ",
                ("strong rhythms", "characteristic", "#c0392b"),
                ", ",
                ("emotional expression", "characteristic", "#c0392b"),
                ", and ",
                ("storytelling", "characteristic", "#c0392b"),
                ". The ",
                ("electric guitar", "instrument", "#2980b9"),
                " plays a significant role in most of these genres, along with ",
                ("drums", "instrument", "#2980b9"),
                " and ",
                ("bass", "instrument", "#2980b9"),
                ". Themes often include ",
                ("love", "theme", "#c0392b"),
                ", ",
                ("hardship", "theme", "#c0392b"),
                ", and ",
                ("social commentary", "theme", "#c0392b"),
                ". Each genre has contributed to the others, with disco incorporating elements of funk and soul, metal evolving from hard rock, and country music influencing early rock and roll. Notable artists span from ",
                ("B.B. King", "artist", "#8e44ad"),
                " (Blues) and ",
                ("Johnny Cash", "artist", "#8e44ad"),
                " (Country) to ",
                ("Donna Summer", "artist", "#8e44ad"),
                " (Disco), ",
                ("Black Sabbath", "artist", "#8e44ad"),
                " (Metal), and ",
                ("The Beatles", "artist", "#8e44ad"),
                " (Rock)."
            )
        elif prediction == 1:
            annotated_text(
                ("Hip Hop, Pop, and Reggae", "genre", "#2d3e50"),
                " are genres that emerged in the ",
                ("mid to late 20th century", "time", "#e67e22"),
                ". Hip Hop originated in ",
                ("New York City", "origin", "#16a085"),
                ", Pop evolved globally, and Reggae came from ",
                ("Jamaica", "origin", "#16a085"),
                ". Despite their distinct origins, these genres share several characteristics. They all have a strong ",
                ("rhythmic focus", "characteristic", "#c0392b"),
                ", emphasize ",
                ("catchy melodies", "characteristic", "#c0392b"),
                ", and often incorporate ",
                ("electronic elements", "characteristic", "#c0392b"),
                ". These genres frequently address ",
                ("social issues", "theme", "#c0392b"),
                ", ",
                ("personal experiences", "theme", "#c0392b"),
                ", and ",
                ("cultural identity", "theme", "#c0392b"),
                ". Common instruments across these genres include ",
                ("synthesizers", "instrument", "#2980b9"),
                ", ",
                ("drum machines", "instrument", "#2980b9"),
                ", and ",
                ("bass guitar", "instrument", "#2980b9"),
                ". All three have had significant ",
                ("cultural impact", "characteristic", "#c0392b"),
                " and ",
                ("global reach", "characteristic", "#c0392b"),
                ", often influencing each other. For example, reggae has influenced hip-hop's use of bass, while pop has borrowed from both genres. These genres have also been at the forefront of ",
                ("music production technology", "characteristic", "#c0392b"),
                ". Notable artists include ",
                ("Tupac Shakur", "artist", "#8e44ad"),
                " (Hip Hop), ",
                ("Michael Jackson", "artist", "#8e44ad"),
                " (Pop), and ",
                ("Bob Marley", "artist", "#8e44ad"),
                " (Reggae), all of whom have had a lasting impact on global music culture."
            )
        elif prediction == 2:
            annotated_text(
                ("Classical", "genre", "#2d3e50"),
                " music spans from the ",
                ("9th century", "time", "#e67e22"),
                " to the present, with significant periods including the ",
                ("Baroque", "period", "#d35400"),
                ", ",
                ("Classical", "period", "#d35400"),
                ", ",
                ("Romantic", "period", "#d35400"),
                ", and ",
                ("Contemporary", "period", "#d35400"),
                ". Rooted in Western liturgical and secular music traditions, it emphasizes ",
                ("formality", "characteristic", "#c0392b"),
                ", complexity, and orchestration, with structured compositions often featuring multiple movements. Key instruments include ",
                ("strings (violin, cello)", "instrument", "#2980b9"),
                ", ",
                ("woodwinds (flute, clarinet)", "instrument", "#2980b9"),
                ", ",
                ("brass (trumpet, trombone)", "instrument", "#2980b9"),
                ", and ",
                ("percussion (timpani)", "instrument", "#2980b9"),
                ". Renowned composers include ",
                ("Johann Sebastian Bach", "artist", "#8e44ad"),
                ", ",
                ("Ludwig van Beethoven", "artist", "#8e44ad"),
                ", and ",
                ("Wolfgang Amadeus Mozart", "artist", "#8e44ad"),
                " whose works continue to resonate in today's musical landscape."
            )
    elif model == "KMeans":
        if prediction == 0:
            annotated_text(
                ("Hip Hop, Pop, and Reggae", "genre", "#2d3e50"),
                " are genres that emerged in the ",
                ("mid to late 20th century", "time", "#e67e22"),
                ". Hip Hop originated in ",
                ("New York City", "origin", "#16a085"),
                ", Pop evolved globally, and Reggae came from ",
                ("Jamaica", "origin", "#16a085"),
                ". These genres share several characteristics, including a strong ",
                ("rhythmic focus", "characteristic", "#c0392b"),
                ", ",
                ("catchy melodies", "characteristic", "#c0392b"),
                ", and frequent use of ",
                ("electronic elements", "characteristic", "#c0392b"),
                ". They often address ",
                ("social issues", "theme", "#c0392b"),
                ", ",
                ("personal experiences", "theme", "#c0392b"),
                ", and ",
                ("cultural identity", "theme", "#c0392b"),
                ". Common instruments include ",
                ("synthesizers", "instrument", "#2980b9"),
                ", ",
                ("drum machines", "instrument", "#2980b9"),
                ", and ",
                ("bass guitar", "instrument", "#2980b9"),
                ". All three have had significant ",
                ("cultural impact", "characteristic", "#c0392b"),
                " and ",
                ("global reach", "characteristic", "#c0392b"),
                ", often influencing each other and leading in ",
                ("music production innovation", "characteristic", "#c0392b"),
                ". Notable artists include ",
                ("Tupac Shakur", "artist", "#8e44ad"),
                " (Hip Hop), ",
                ("Michael Jackson", "artist", "#8e44ad"),
                " (Pop), and ",
                ("Bob Marley", "artist", "#8e44ad"),
                " (Reggae)."
            )
        elif prediction == 1:
            annotated_text(
                ("Classical", "genre", "#2d3e50"),
                " music spans from the ",
                ("9th century", "time", "#e67e22"),
                " to the present, with significant periods including the ",
                ("Baroque", "period", "#d35400"),
                ", ",
                ("Classical", "period", "#d35400"),
                ", ",
                ("Romantic", "period", "#d35400"),
                ", and ",
                ("Contemporary", "period", "#d35400"),
                ". Rooted in Western liturgical and secular music traditions, it emphasizes ",
                ("formality", "characteristic", "#c0392b"),
                ", ",
                ("complexity", "characteristic", "#c0392b"),
                ", and ",
                ("orchestration", "characteristic", "#c0392b"),
                ", with structured compositions often featuring multiple movements. Key instruments include ",
                ("strings (violin, cello)", "instrument", "#2980b9"),
                ", ",
                ("woodwinds (flute, clarinet)", "instrument", "#2980b9"),
                ", ",
                ("brass (trumpet, trombone)", "instrument", "#2980b9"),
                ", and ",
                ("percussion (timpani)", "instrument", "#2980b9"),
                ". Renowned composers include ",
                ("Johann Sebastian Bach", "artist", "#8e44ad"),
                ", ",
                ("Ludwig van Beethoven", "artist", "#8e44ad"),
                ", and ",
                ("Wolfgang Amadeus Mozart", "artist", "#8e44ad"),
                " whose works continue to resonate in today's musical landscape."
            )
        elif prediction == 2:
            annotated_text(
                ("Metal and Rock", "genre", "#2d3e50"),
                " are closely related genres that emerged in the ",
                ("1950s to 1970s", "time", "#e67e22"),
                ". Both evolved from ",
                ("blues", "influence", "#d35400"),
                " and ",
                ("rhythm and blues", "influence", "#d35400"),
                ". These genres are characterized by their ",
                ("heavy use of electric guitars", "characteristic", "#c0392b"),
                ", ",
                ("strong rhythms", "characteristic", "#c0392b"),
                ", and ",
                ("energetic performances", "characteristic", "#c0392b"),
                ". Metal often features more ",
                ("aggressive sounds", "characteristic", "#c0392b"),
                " and ",
                ("complex compositions", "characteristic", "#c0392b"),
                ", while rock spans a wide range of styles. Common instruments include ",
                ("electric guitar", "instrument", "#2980b9"),
                ", ",
                ("bass guitar", "instrument", "#2980b9"),
                ", and ",
                ("drums", "instrument", "#2980b9"),
                ". Both genres often explore themes of ",
                ("rebellion", "theme", "#c0392b"),
                ", ",
                ("social issues", "theme", "#c0392b"),
                ", and ",
                ("personal freedom", "theme", "#c0392b"),
                ". Influential artists span from rock icons like ",
                ("The Beatles", "artist", "#8e44ad"),
                " and ",
                ("Led Zeppelin", "artist", "#8e44ad"),
                " to metal pioneers such as ",
                ("Black Sabbath", "artist", "#8e44ad"),
                " and ",
                ("Metallica", "artist", "#8e44ad"),
                "."
            )
        elif prediction == 3:
            annotated_text(
                ("Blues, Country, Disco, and Jazz", "genre", "#2d3e50"),
                " are genres that developed primarily in the ",
                ("United States", "origin", "#16a085"),
                " from the ",
                ("late 19th to mid-20th century", "time", "#e67e22"),
                ". They share roots in ",
                ("African American musical traditions", "influence", "#d35400"),
                " and often feature ",
                ("emotional depth", "characteristic", "#c0392b"),
                " and ",
                ("improvisation", "characteristic", "#c0392b"),
                ". Blues and Country often address themes of ",
                ("love", "theme", "#c0392b"),
                ", ",
                ("loss", "theme", "#c0392b"),
                ", and ",
                ("hardship", "theme", "#c0392b"),
                ". Jazz and Disco emphasize ",
                ("complex rhythms", "characteristic", "#c0392b"),
                " and are closely tied to ",
                ("dance culture", "characteristic", "#c0392b"),
                ". These genres have significantly influenced each other and modern popular music. Common instruments across these genres include ",
                ("guitar", "instrument", "#2980b9"),
                ", ",
                ("piano", "instrument", "#2980b9"),
                ", ",
                ("bass", "instrument", "#2980b9"),
                ", and various ",
                ("horns", "instrument", "#2980b9"),
                ". Notable artists include ",
                ("B.B. King", "artist", "#8e44ad"),
                " (Blues), ",
                ("Johnny Cash", "artist", "#8e44ad"),
                " (Country), ",
                ("Donna Summer", "artist", "#8e44ad"),
                " (Disco), and ",
                ("Miles Davis", "artist", "#8e44ad"),
                " (Jazz)."
            )
    elif model == "GMM":
        if prediction == 0:
            annotated_text(
                ("Hip Hop, Jazz, Pop, and Reggae", "genre", "#2d3e50"),
                " are diverse genres that emerged between the ",
                ("early 20th and late 20th century", "time", "#e67e22"),
                ". While originating in different locations - Hip Hop in ",
                ("New York City", "origin", "#16a085"),
                ", Jazz in ",
                ("New Orleans", "origin", "#16a085"),
                ", Pop globally, and Reggae in ",
                ("Jamaica", "origin", "#16a085"),
                " - these genres share several characteristics. They all have a strong ",
                ("rhythmic focus", "characteristic", "#c0392b"),
                ", emphasize ",
                ("improvisation", "characteristic", "#c0392b"),
                " to varying degrees, and often incorporate ",
                ("syncopation", "characteristic", "#c0392b"),
                ". These genres frequently address ",
                ("social issues", "theme", "#c0392b"),
                ", ",
                ("personal experiences", "theme", "#c0392b"),
                ", and ",
                ("cultural identity", "theme", "#c0392b"),
                ". Common instruments across these genres include ",
                ("drums", "instrument", "#2980b9"),
                ", ",
                ("bass", "instrument", "#2980b9"),
                ", and various ",
                ("horns", "instrument", "#2980b9"),
                ", with modern forms often incorporating ",
                ("electronic instruments", "instrument", "#2980b9"),
                ". All four have had significant ",
                ("cultural impact", "characteristic", "#c0392b"),
                " and ",
                ("global reach", "characteristic", "#c0392b"),
                ", often influencing each other. Notable artists include ",
                ("Tupac Shakur", "artist", "#8e44ad"),
                " (Hip Hop), ",
                ("Miles Davis", "artist", "#8e44ad"),
                " (Jazz), ",
                ("Michael Jackson", "artist", "#8e44ad"),
                " (Pop), and ",
                ("Bob Marley", "artist", "#8e44ad"),
                " (Reggae)."
            )
        elif prediction == 1:
            annotated_text(
                ("Blues, Country, Disco, Metal, and Rock", "genre", "#2d3e50"),
                " are genres that developed primarily in the ",
                ("United States", "origin", "#16a085"),
                " from the ",
                ("late 19th to late 20th century", "time", "#e67e22"),
                ". Despite their differences, these genres share some common elements. They all have roots in ",
                ("earlier American musical traditions", "influence", "#d35400"),
                ", particularly ",
                ("blues", "influence", "#d35400"),
                ". These genres often feature ",
                ("strong rhythms", "characteristic", "#c0392b"),
                ", ",
                ("emotional expression", "characteristic", "#c0392b"),
                ", and ",
                ("storytelling", "characteristic", "#c0392b"),
                ". The ",
                ("electric guitar", "instrument", "#2980b9"),
                " plays a significant role in most of these genres, along with ",
                ("drums", "instrument", "#2980b9"),
                " and ",
                ("bass", "instrument", "#2980b9"),
                ". Themes often include ",
                ("love", "theme", "#c0392b"),
                ", ",
                ("hardship", "theme", "#c0392b"),
                ", ",
                ("rebellion", "theme", "#c0392b"),
                ", and ",
                ("social commentary", "theme", "#c0392b"),
                ". Each genre has contributed to the evolution of the others, creating a rich tapestry of American music. Notable artists span from ",
                ("B.B. King", "artist", "#8e44ad"),
                " (Blues), ",
                ("Johnny Cash", "artist", "#8e44ad"),
                " (Country), ",
                ("Donna Summer", "artist", "#8e44ad"),
                " (Disco), ",
                ("Black Sabbath", "artist", "#8e44ad"),
                " (Metal), to ",
                ("The Beatles", "artist", "#8e44ad"),
                " (Rock)."
            )
        elif prediction == 2:
            annotated_text(
                ("Classical", "genre", "#2d3e50"),
                " music spans from the ",
                ("9th century", "time", "#e67e22"),
                " to the present, with significant periods including the ",
                ("Baroque", "period", "#d35400"),
                ", ",
                ("Classical", "period", "#d35400"),
                ", ",
                ("Romantic", "period", "#d35400"),
                ", and ",
                ("Contemporary", "period", "#d35400"),
                ". Rooted in Western liturgical and secular music traditions, it emphasizes ",
                ("formality", "characteristic", "#c0392b"),
                ", ",
                ("complexity", "characteristic", "#c0392b"),
                ", and ",
                ("orchestration", "characteristic", "#c0392b"),
                ", with structured compositions often featuring multiple movements. Key instruments include ",
                ("strings (violin, cello)", "instrument", "#2980b9"),
                ", ",
                ("woodwinds (flute, clarinet)", "instrument", "#2980b9"),
                ", ",
                ("brass (trumpet, trombone)", "instrument", "#2980b9"),
                ", and ",
                ("percussion (timpani)", "instrument", "#2980b9"),
                ". Renowned composers include ",
                ("Johann Sebastian Bach", "artist", "#8e44ad"),
                ", ",
                ("Ludwig van Beethoven", "artist", "#8e44ad"),
                ", and ",
                ("Wolfgang Amadeus Mozart", "artist", "#8e44ad"),
                " whose works continue to resonate in today's musical landscape."
            )
        elif prediction == 3:
            annotated_text(
                ("Unidentified Genre", "genre", "#2d3e50"),
                ". The algorithm was unable to categorize this music into a specific genre. This could be due to several factors:",
                ("\n\n1. ", "characteristic", "#c0392b"),
                "The music may blend elements from multiple genres, making it difficult to classify.",
                ("\n2. ", "characteristic", "#c0392b"),
                "It could be an emerging or niche genre not yet recognized by the algorithm.",
                ("\n3. ", "characteristic", "#c0392b"),
                "The music might have unique characteristics that don't align with the algorithm's known categories.",
                ("\n4. ", "characteristic", "#c0392b"),
                "There may be insufficient data in the algorithm's training set to accurately classify this type of music.",
                ("\n\nFurther analysis ", "characteristic", "#c0392b"),
                "by music experts or more advanced algorithms might be needed to properly categorize this music. This 'unidentified' category highlights the diverse and evolving nature of music, where new styles and fusions continually emerge."
            )
    elif model == "Birch":
        if prediction == 0:
                annotated_text(
                    ("Classical", "genre", "#2d3e50"),
                    " music spans from the ",
                    ("9th century", "time", "#e67e22"),
                    " to the present, with significant periods including the ",
                    ("Baroque", "period", "#d35400"),
                    ", ",
                    ("Classical", "period", "#d35400"),
                    ", ",
                    ("Romantic", "period", "#d35400"),
                    ", and ",
                    ("Contemporary", "period", "#d35400"),
                    ". Rooted in Western liturgical and secular music traditions, it emphasizes ",
                    ("formality", "characteristic", "#c0392b"),
                    ", ",
                    ("complexity", "characteristic", "#c0392b"),
                    ", and ",
                    ("orchestration", "characteristic", "#c0392b"),
                    ", with structured compositions often featuring multiple movements. Key instruments include ",
                    ("strings (violin, cello)", "instrument", "#2980b9"),
                    ", ",
                    ("woodwinds (flute, clarinet)", "instrument", "#2980b9"),
                    ", ",
                    ("brass (trumpet, trombone)", "instrument", "#2980b9"),
                    ", and ",
                    ("percussion (timpani)", "instrument", "#2980b9"),
                    ". Renowned composers include ",
                    ("Johann Sebastian Bach", "artist", "#8e44ad"),
                    ", ",
                    ("Ludwig van Beethoven", "artist", "#8e44ad"),
                    ", and ",
                    ("Wolfgang Amadeus Mozart", "artist", "#8e44ad"),
                    " whose works continue to resonate in today's musical landscape."
                )
        elif prediction in [1, 2, 3]:
                annotated_text(
                    ("Unidentified Genre Cluster", "genre", "#2d3e50"),
                    f" {prediction}",
                    ". The algorithm was unable to categorize this music into a specific genre. This could be due to several factors:",
                    ("\n\n1. ", "characteristic", "#c0392b"),
                    "The music in this cluster may blend elements from multiple genres, making it difficult to classify definitively.",
                    ("\n2. ", "characteristic", "#c0392b"),
                    "It could represent emerging or niche genres not yet recognized by the algorithm.",
                    ("\n3. ", "characteristic", "#c0392b"),
                    "The music might have unique characteristics that don't align with the algorithm's known categories.",
                    ("\n4. ", "characteristic", "#c0392b"),
                    "There may be insufficient data in the algorithm's training set to accurately classify this type of music.",
                    ("\n\nPossible interpretations", "characteristic", "#c0392b"),
                    " for these unidentified clusters could include:",
                    ("\n\n- Cluster 1: ", "interpretation", "#d35400"),
                    "This might represent a blend of popular music genres, possibly including elements of ",
                    ("rock", "genre", "#2d3e50"),
                    ", ",
                    ("pop", "genre", "#2d3e50"),
                    ", and ",
                    ("electronic music", "genre", "#2d3e50"),
                    ".",
                    ("\n- Cluster 2: ", "interpretation", "#d35400"),
                    "This could be a cluster of various ",
                    ("world music", "genre", "#2d3e50"),
                    " genres, incorporating diverse cultural influences and instruments.",
                    ("\n- Cluster 3: ", "interpretation", "#d35400"),
                    "This might represent experimental or avant-garde music that defies traditional genre classifications.",
                    ("\n\nFurther analysis ", "characteristic", "#c0392b"),
                    "by music experts or more advanced algorithms might be needed to properly categorize these music clusters. The presence of multiple unidentified categories highlights the diverse and evolving nature of music, where new styles and fusions continually emerge, challenging traditional classification systems."
                )
        elif model == "DBSCAN":
            if prediction == 0:
                annotated_text(
                    ("Single Music Cluster", "genre", "#2d3e50"),
                    ". The algorithm has categorized all music into a single cluster. This outcome is due to the ",
                    ("algorithm's limitation", "characteristic", "#c0392b"),
                    " of being able to identify only one cluster. This result does not necessarily reflect the true diversity of the music analyzed, but rather the constraints of the classification method. Here are some points to consider:",
                    ("\n\n1. ", "limitation", "#c0392b"),
                    "Algorithm Design: The algorithm is designed or configured to output only one cluster, regardless of the input data's diversity.",
                    ("\n2. ", "limitation", "#c0392b"),
                    "Oversimplification: This approach drastically oversimplifies the complex landscape of musical genres and styles.",
                    ("\n3. ", "limitation", "#c0392b"),
                    "Loss of Nuance: Important distinctions between different types of music are not captured in this single-cluster model.",
                    ("\n4. ", "limitation", "#c0392b"),
                    "Potential Bias: The single cluster might be biased towards the most common or dominant type of music in the dataset.",
                    ("\n\nImplications", "characteristic", "#c0392b"),
                    " of this single-cluster result:",
                    ("\n\n- ", "implication", "#d35400"),
                    "Limited Utility: This classification provides very little practical information about the nature or diversity of the music analyzed.",
                    ("\n- ", "implication", "#d35400"),
                    "Need for Refinement: It clearly indicates that the algorithm needs significant improvement to be useful for music classification.",
                    ("\n- ", "implication", "#d35400"),
                    "Baseline for Comparison: This could serve as a baseline against which more sophisticated multi-cluster algorithms can be compared.",
                    ("\n\nPossible next steps", "characteristic", "#c0392b"),
                    " to improve the analysis:",
                    ("\n\n1. ", "step", "#16a085"),
                    "Algorithm Enhancement: Modify the algorithm to allow for multiple clusters, enabling it to capture more nuanced distinctions in music.",
                    ("\n2. ", "step", "#16a085"),
                    "Feature Engineering: Develop more sophisticated features for the algorithm to consider, which might help it distinguish between different types of music.",
                    ("\n3. ", "step", "#16a085"),
                    "Human Oversight: Incorporate human expertise to validate and refine the algorithm's classifications.",
                    ("\n4. ", "step", "#16a085"),
                    "Hybrid Approach: Consider combining this algorithm with other classification methods to create a more comprehensive analysis system."
                )
            else:
                annotated_text(
                    ("Error", "status", "#e74c3c"),
                    ": The algorithm is designed to output only one cluster (cluster 0). This result (cluster ",
                    (f"{prediction}", "cluster", "#3498db"),
                    ") is unexpected and may indicate a malfunction in the algorithm or data processing pipeline."
                )



def get_mp3_metadata(file_path):
    try:
        audio = MP3(file_path, ID3=EasyID3)
        title = audio.get('title', ['Unknown Title'])[0]
        artists = audio.get('artist', ['Unknown Artist'])[0]
        return title, artists
    except Exception as e:
        print(f"Error retrieving metadata for {file_path}: {e}")
        return None, None

def show_recommendations(predictions, model):
    st.title("Songs of the same genre:")
    if model == "Agglomerative":
        genres = [["hip-hop", "pop", "reggae"],
                  ["classical"],
                  ["metal", "rock"],
                  ["blues", "country", "disco", "jazz"]]
    elif model == "Mean Shift":
        genres = [["blues", "country", "disco", "metal", "rock"],
                  ["hip-hop", "pop", "reggae"],
                  ["classical", "jazz"]]
    elif model == "KMeans":
        genres = [["hip-hop", "pop", "reggae"],
                  ["classical"],
                  ["metal", "rock"],
                  ["blues", "country", "disco", "jazz"]]
    elif model == "GMM":
        genres = [["hip-hop", "jazz", "pop", "reggae"],
                  ["blues", "country", "disco", "metal", "rock"],
                  ["classical"],
                  ["unidentified"]]
    elif model == "Birch":
        genres = [["classical"],
                  ["unidentified"],
                  ["unidentified"],
                  ["unidentified"]]
    elif model == "DBSCAN":
        genres = [["unidentified"]]

    predicted_genres = genres[predictions]
    
    # Remove "unidentified" from the list of genres
    predicted_genres = [genre for genre in predicted_genres if genre != "unidentified"]
    
    if not predicted_genres:
        st.write("No identified genres to recommend songs from.")
        return

    # Randomly choose 5 numbers from 1 to 500
    random_numbers = random.sample(range(1, 501), 5)
    for i in range(5):
        # Randomly choose a genre for each iteration
        chosen_genre = random.choice(predicted_genres)
        folder_path = f"Audio_Spotify/{chosen_genre}/"
        
        file_path = folder_path + str(random_numbers[i]) + ".mp3"
        title, artists = get_mp3_metadata(file_path)
        colour_palettes = [["#522258", "#8C3061", "#C63C51"],
                           ["#3E3232", "#503C3C", "#7E6363"],
                           ["#212A3E", "#394867", "#9BA4B5"],
                           ["#3A3845", "#826F66", "#C69B7B"],
                           ["#632626", "#9D5353", "#Bf8B67"]]
        annotated_text(
            (f"{i+1}. ", "" , colour_palettes[i%5][0]),
            (f"{title}", "song", colour_palettes[i%5][1]),
            (f" by {artists}", "artist", colour_palettes[i%5][2]),
            (f" ({chosen_genre})", "genre", "#4A4A4A")
        )
        st.audio(file_path, format="audio/mp3")

def plot_kmeans(umap, new_umap):
    # Load the KMeans model
    kmeans = joblib.load("models/kmeans_model.pkl")

    labels = kmeans.fit_predict(umap)
    centers = kmeans.cluster_centers_

    metrics_df = pd.read_csv("Metrics.csv")
    
    df = pd.DataFrame({
        'UMAP Dimension 1': umap[:, 0],
        'UMAP Dimension 2': umap[:, 1],
        'Cluster': labels
    })

    metrics = {
        'Silhouette Score': metrics_df[metrics_df['model'] == 'K-Means-UMAP (Fine Tuned)']['Silhouette Score'].values[0],
        'Davies-Bouldin Score': metrics_df[metrics_df['model'] == 'K-Means-UMAP (Fine Tuned)']['Davies-Bouldin Index'].values[0],
        'Calinski-Harabasz Score': metrics_df[metrics_df['model'] == 'K-Means-UMAP (Fine Tuned)']['Calinski-Harabasz Index'].values[0],
        'Inertia': metrics_df[metrics_df['model'] == 'K-Means-UMAP (Fine Tuned)']['Inertia'].values[0]
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['UMAP Dimension 1'],
                         df['UMAP Dimension 2'],
                         c=df['Cluster'], cmap='viridis', alpha=0.5)
    
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X',
               edgecolor='black', label='Centroids')
    
    # Plot the new prediction
    # Ensure new_umap is treated as a 1x2 array
    new_umap = np.array(new_umap).reshape(1, -1)
    ax.scatter(new_umap[0, 0], new_umap[0, 1], c='yellow', s=200, marker='*',
               edgecolor='black', label='New Prediction')
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(np.unique(labels))))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                          markersize=10, label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='X', color='red',
                              markersize=10, label='Centroids', markeredgecolor='black'))
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='*', color='yellow',
                              markersize=10, label='New Prediction', markeredgecolor='black'))
    
    ax.legend(handles=handles, title='Cluster Labels', loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_title('UMAP with KMeans Clustering (k=4), Centroids, and New Prediction')

    metrics_text = f"Evaluation Metrics:\n"
    metrics_text += f"Silhouette Score: {metrics['Silhouette Score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Index: {metrics['Davies-Bouldin Score']:.4f}\n"
    metrics_text += f"Inertia: {metrics['Inertia']:.4f}"

    plt.text(1.02, 0.5, metrics_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_agglomerative(umap, new_umap):
    # Load the Agglomerative model
    agglomerative = joblib.load("models/agglomerative_model.pkl")

    labels = agglomerative.fit_predict(umap)

    metrics_df = pd.read_csv("Metrics.csv")
    
    df = pd.DataFrame({
        'UMAP Dimension 1': umap[:, 0],
        'UMAP Dimension 2': umap[:, 1],
        'Cluster': labels
    })

    metrics = {
        'Silhouette Score': metrics_df[metrics_df['model'] == 'Agglomerative-UMAP (Fine Tuned)']['Silhouette Score'].values[0],
        'Davies-Bouldin Score': metrics_df[metrics_df['model'] == 'Agglomerative-UMAP (Fine Tuned)']['Davies-Bouldin Index'].values[0],
        'Calinski-Harabasz Score': metrics_df[metrics_df['model'] == 'Agglomerative-UMAP (Fine Tuned)']['Calinski-Harabasz Index'].values[0]
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['UMAP Dimension 1'],
                         df['UMAP Dimension 2'],
                         c=df['Cluster'], cmap='viridis', alpha=0.5)
    
    # Plot the new prediction
    # Ensure new_umap is treated as a 1x2 array
    new_umap = np.array(new_umap).reshape(1, -1)
    ax.scatter(new_umap[0, 0], new_umap[0, 1], c='yellow', s=200, marker='*',
               edgecolor='black', label='New Prediction')
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(np.unique(labels))))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                          markersize=10, label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='*', color='yellow',
                              markersize=10, label='New Prediction', markeredgecolor='black'))
    
    ax.legend(handles=handles, title='Cluster Labels', loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_title('UMAP with Agglomerative Clustering, and New Prediction')

    metrics_text = f"Evaluation Metrics:\n"
    metrics_text += f"Silhouette Score: {metrics['Silhouette Score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Index: {metrics['Davies-Bouldin Score']:.4f}"

    plt.text(1.02, 0.5, metrics_text
                , transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_gmm(umap, new_umap):
    
    gmm = joblib.load("models/gmm_model.pkl")
    labels = gmm.fit_predict(umap)
    centers = gmm.means_

    metrics_df = pd.read_csv("Metrics.csv")

    df = pd.DataFrame({
        'UMAP Dimension 1': umap[:, 0],
        'UMAP Dimension 2': umap[:, 1],
        'Cluster': labels
    })

    metrics = {
        'Silhouette Score': metrics_df[metrics_df['model'] == 'GMM-UMAP (Fine Tuned)']['Silhouette Score'].values[0],
        'Davies-Bouldin Score': metrics_df[metrics_df['model'] == 'GMM-UMAP (Fine Tuned)']['Davies-Bouldin Index'].values[0],
        'Calinski-Harabasz Score': metrics_df[metrics_df['model'] == 'GMM-UMAP (Fine Tuned)']['Calinski-Harabasz Index'].values[0]
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['UMAP Dimension 1'],
                            df['UMAP Dimension 2'],
                            c=df['Cluster'], cmap='viridis', alpha=0.5)
    
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X',
                edgecolor='black', label='Centroids')
    
    # Plot the new prediction
    # Ensure new_umap is treated as a 1x2 array
    new_umap = np.array(new_umap).reshape(1, -1)
    ax.scatter(new_umap[0, 0], new_umap[0, 1], c='yellow', s=200, marker='*',
                edgecolor='black', label='New Prediction')
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(np.unique(labels))))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                            markersize=10, label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='X', color='red',
                            markersize=10, label='Centroids', markeredgecolor='black'))
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='*', color='yellow',
                            markersize=10, label='New Prediction', markeredgecolor='black'))
    
    ax.legend(handles=handles, title='Cluster Labels', loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_title('UMAP with GMM Clustering, Centroids, and New Prediction')

    metrics_text = f"Evaluation Metrics:\n"
    metrics_text += f"Silhouette Score: {metrics['Silhouette Score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Index: {metrics['Davies-Bouldin Score']:.4f}"

    plt.text(1.02, 0.5, metrics_text
             , transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_mean_shift(tsne, new_tsne):
    mean_shift = joblib.load("models/meanshift_model.pkl")
    labels = mean_shift.fit_predict(tsne)
    centers = mean_shift.cluster_centers_

    metrics_df = pd.read_csv("Metrics.csv")

    df = pd.DataFrame({
        'TSNE Dimension 1': tsne[:, 0],
        'TSNE Dimension 2': tsne[:, 1],
        'Cluster': labels
    })

    metrics = {
        'Silhouette Score': metrics_df[metrics_df['model'] == 'Mean Shift-TSNE(Fine Tuned)']['Silhouette Score'].values[0],
        'Davies-Bouldin Score': metrics_df[metrics_df['model'] == 'Mean Shift-TSNE(Fine Tuned)']['Davies-Bouldin Index'].values[0],
        'Calinski-Harabasz Score': metrics_df[metrics_df['model'] == 'Mean Shift-TSNE(Fine Tuned)']['Calinski-Harabasz Index'].values[0]
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['TSNE Dimension 1'],
                            df['TSNE Dimension 2'],
                            c=df['Cluster'], cmap='viridis', alpha=0.5)
    
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X',
                edgecolor='black', label='Centroids')
    
    # Plot the new prediction
    # Ensure new_tsne is treated as a 1x2 array
    new_tsne = np.array(new_tsne).reshape(1, -1)
    ax.scatter(new_tsne[0, 0], new_tsne[0, 1], c='yellow', s=200, marker='*',
                edgecolor='black', label='New Prediction')
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(np.unique(labels))))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                            markersize=10, label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='X', color='red',
                            markersize=10, label='Centroids', markeredgecolor='black'))
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='*', color='yellow',
                            markersize=10, label='New Prediction', markeredgecolor='black'))
    
    ax.legend(handles=handles, title='Cluster Labels', loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_title('TSNE with Mean Shift Clustering, Centroids, and New Prediction')

    metrics_text = f"Evaluation Metrics:\n"
    metrics_text += f"Silhouette Score: {metrics['Silhouette Score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Index: {metrics['Davies-Bouldin Score']:.4f}"

    plt.text(1.02, 0.5, metrics_text
                , transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_birch(umap, new_umap):
    birch = joblib.load("models/birch_model.pkl")
    labels = birch.fit_predict(umap)

    metrics_df = pd.read_csv("Metrics.csv")

    df = pd.DataFrame({
        'UMAP Dimension 1': umap[:, 0],
        'UMAP Dimension 2': umap[:, 1],
        'Cluster': labels
    })

    metrics = {
        'Silhouette Score': metrics_df[metrics_df['model'] == 'Birch-UMAP(Fine Tuned)']['Silhouette Score'].values[0],
        'Davies-Bouldin Score': metrics_df[metrics_df['model'] == 'Birch-UMAP(Fine Tuned)']['Davies-Bouldin Index'].values[0],
        'Calinski-Harabasz Score': metrics_df[metrics_df['model'] == 'Birch-UMAP(Fine Tuned)']['Calinski-Harabasz Index'].values[0]
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['UMAP Dimension 1'],
                            df['UMAP Dimension 2'],
                            c=df['Cluster'], cmap='viridis', alpha=0.5)
    
    # Plot the new prediction
    # Ensure new_umap is treated as a 1x2 array
    new_umap = np.array(new_umap).reshape(1, -1)
    ax.scatter(new_umap[0, 0], new_umap[0, 1], c='yellow', s=200, marker='*',
                edgecolor='black', label='New Prediction')
    
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(np.unique(labels))))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                            markersize=10, label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='X', color='red',
                            markersize=10, label='Centroids', markeredgecolor='black'))
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='*', color='yellow',
                            markersize=10, label='New Prediction', markeredgecolor='black'))
    
    ax.legend(handles=handles, title='Cluster Labels', loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')

    ax.set_title('UMAP with Birch Clustering, Centroids, and New Prediction')

    metrics_text = f"Evaluation Metrics:\n"
    metrics_text += f"Silhouette Score: {metrics['Silhouette Score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Index: {metrics['Davies-Bouldin Score']:.4f}"

    plt.text(1.02, 0.5, metrics_text
                , transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# def ground_truth_table(model):
#      if model != "DBSCAN":
#         st.subheader("Ground Truth Table")
#         st.image('clustering_ground_truth/' + model + '.png')

    

def plot_dbscan(pca, new_pca):
    dbscan = joblib.load("models/dbscan_model.pkl")
    labels = dbscan.fit_predict(pca)

    metrics_df = pd.read_csv("Metrics.csv")

    df = pd.DataFrame({
        'PCA Dimension 1': pca[:, 0],
        'PCA Dimension 2': pca[:, 1],
        'Cluster': labels
    })

    metrics = {
        'Silhouette Score': metrics_df[metrics_df['model'] == 'DBSCAN-PCA(Fine Tuned)']['Silhouette Score'].values[0],
        'Davies-Bouldin Score': metrics_df[metrics_df['model'] == 'DBSCAN-PCA(Fine Tuned)']['Davies-Bouldin Index'].values[0],
        'Calinski-Harabasz Score': metrics_df[metrics_df['model'] == 'DBSCAN-PCA(Fine Tuned)']['Calinski-Harabasz Index'].values[0]
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['PCA Dimension 1'],
                            df['PCA Dimension 2'],
                            c=df['Cluster'], cmap='viridis', alpha=0.5)
    
    # Plot the new prediction
    # Ensure new_pca is treated as a 1x2 array
    new_pca = np.array(new_pca).reshape(1, -1)
    ax.scatter(new_pca[0, 0], new_pca[0, 1], c='yellow', s=200, marker='*', edgecolor='black', label='New Prediction')

    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(np.unique(labels))))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                            markersize=10, label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
    handles.append(plt.Line2D([0], [0], linestyle='None', marker='*', color='yellow',
                            markersize=10, label='New Prediction', markeredgecolor='black'))
    
    ax.legend(handles=handles, title='Cluster Labels', loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('PCA with DBSCAN Clustering, and New Prediction')

    metrics_text = f"Evaluation Metrics:\n"
    metrics_text += f"Silhouette Score: {metrics['Silhouette Score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Index: {metrics['Calinski-Harabasz Score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Index: {metrics['Davies-Bouldin Score']:.4f}"

    plt.text(1.02, 0.5, metrics_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

# Cache data loading
@st.cache_data
def load_data():
    return pd.read_csv("dataset_30s.csv")

# Cache model loading
@st.cache_resource
def load_models():
    return {
        'scaler': joblib.load("models/scaler.pkl"),
        'pca': joblib.load("models/pca_model.pkl")
    }



def main():
    page = st.sidebar.selectbox("Select Page", ["Prediction", "Dashboard"])
    
    if page == "Prediction":
        st.title("Music Genre Clustering")
        proceed = False
        with st.sidebar:
            music_file = st.file_uploader("Upload Audio File", type="mp3")
            if music_file:
                y, sr, audio_features = load_and_extract_audio_features(music_file)
                models = st.selectbox("Select Model", ["KMeans", "Agglomerative", "GMM", "Mean Shift", "Birch", "DBSCAN"])
                if models == "KMeans":
                    predict_button = st.button("Predict Genre")
                    if predict_button:
                        predictions, labels, umap, new_umap= make_predictions(audio_features, "KMeans")
                        proceed = True
                elif models == "Agglomerative":
                    predict_button = st.button("Predict Genre")
                    if predict_button:
                        predictions , labels, umap, new_umap= make_predictions(audio_features, "Agglomerative")
                        proceed = True
                elif models == "GMM":
                    predict_button = st.button("Predict Genre")
                    if predict_button:
                        predictions, labels, umap, new_umap = make_predictions(audio_features, "GMM")
                        proceed = True
                elif models == "Mean Shift":
                    predict_button = st.button("Predict Genre")
                    if predict_button:
                        predictions, labels, tsne, new_tsne = make_predictions(audio_features, "Mean Shift")
                        proceed = True
                elif models == "Birch":
                    predict_button = st.button("Predict Genre")
                    if predict_button:
                        predictions, labels, umap, new_umap = make_predictions(audio_features, "Birch")
                        proceed = True
                elif models == "DBSCAN":
                    predict_button = st.button("Predict Genre")
                    if predict_button:
                        predictions, labels, pca, new_pca = make_predictions(audio_features, "DBSCAN")
                        proceed = True
                else:
                    st.info("Please select a model to predict the music genre.")
                    predictions = None
                
            else:
                st.info("Please upload an audio file to get started.")
                y, sr, audio_features = None, None, None  

        if proceed:
            # Create time array for the waveform
            time_array = np.linspace(0, len(y) / sr, num=len(y))
                
            # Create a Plotly figure for the waveform
            fig_waveform = go.Figure()
                
            # Add waveform trace
            fig_waveform.add_trace(go.Scatter(x=time_array, y=y, mode='lines', name='Waveform'))
                
            # Update layout for the waveform
            fig_waveform.update_layout(title='Waveform',
                                                xaxis_title='Time (s)',
                                                yaxis_title='Amplitude',
                                                height=400)

            # Display the waveform plot in Streamlit
            st.plotly_chart(fig_waveform, use_container_width=True)
            plot_spectral_centroid(y, sr)
            plot_spectral_rolloff(y, sr)
            plot_stft(y, sr)


            audio = st.audio(music_file, start_time=0)
            
            if models == "KMeans":
                plot_kmeans(umap, new_umap)
            elif models == "Agglomerative":
                plot_agglomerative(umap, new_umap)
            elif models == "GMM":
                plot_gmm(umap, new_umap)
            elif models == "Mean Shift":
                plot_mean_shift(tsne, new_tsne)
            elif models == "Birch":
                plot_birch(umap, new_umap)
            elif models == "DBSCAN":
                plot_dbscan(pca, new_pca)

            
            # ground_truth_table(models)
            show_genre(predictions, models)
            show_recommendations(predictions, models)

    elif page == "Dashboard":
        # Load data and models
        df = load_data()
        models = load_models()
        st.title("Music Genre Clustering Dashboard")
        
        st.write("Data Preview:", df.head())
        
        # Preprocess the data
        scaled_features = models['scaler'].transform(df)
        
        # Cache dimensionality reduction
        @st.cache_data
        def perform_dim_reduction(scaled_features):
            import umap
            umap_model_2d = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.0, random_state=42)
            umap_model_3d = umap.UMAP(n_components=3, n_neighbors=5, min_dist=0.0, random_state=42)
            tsne_model_2d = TSNE(n_components=2, perplexity=50, learning_rate=1000, n_iter=1000)
            tsne_model_3d = TSNE(n_components=3, perplexity=50, learning_rate=1000, n_iter=1000)
            pca_model_2d = models['pca']
            pca_model_3d = PCA(n_components=3)
            pca_model_3d.fit(scaled_features)
            return {
                'UMAP_2D': umap_model_2d.fit_transform(scaled_features),
                'UMAP_3D': umap_model_3d.fit_transform(scaled_features),
                't-SNE_2D': tsne_model_2d.fit_transform(scaled_features),
                't-SNE_3D': tsne_model_3d.fit_transform(scaled_features),
                'PCA_2D': pca_model_2d.transform(scaled_features),
                'PCA_3D': pca_model_3d.transform(scaled_features)
            }
        
        dim_reduction_results = perform_dim_reduction(scaled_features)
        # Your existing code for data exploration, visualization, model performance, etc.

        if st.checkbox("Show Dimension Reduction Before Clustering"):
            st.markdown("## Dimension Reduction Before Clustering")
            
            dim_reduction_technique = st.selectbox("Select Dimensionality Reduction Technique (Pre-Clustering):", 
                                                   ['PCA', 'UMAP', 't-SNE'], key="pre_clustering_technique")
            dim_reduction_dimension = st.selectbox("Select Dimension (Pre-Clustering):", ['2D', '3D'], key="pre_clustering_dimension")
            
            result_key = f"{dim_reduction_technique}_{dim_reduction_dimension}"
            result = dim_reduction_results[result_key]
            
            trustworthiness_score = trustworthiness(scaled_features, result, n_neighbors=5)
            
            # Display trustworthiness
            st.markdown("### Trustworthiness")
            st.metric("Trustworthiness", f"{trustworthiness_score:.2f}", help="Higher is better")
            
            # Plot the dimension reduction result
            if dim_reduction_dimension == '2D':
                fig = px.scatter(x=result[:, 0], y=result[:, 1], 
                                 title=f"Dimension Reduction ({dim_reduction_technique} - {dim_reduction_dimension})",
                                 labels={'color': 'Cluster'},
                                 color_continuous_scale='Rainbow')
            else:
                fig = px.scatter_3d(x=result[:, 0], y=result[:, 1], z=result[:, 2],
                                    title=f"Dimension Reduction ({dim_reduction_technique} - {dim_reduction_dimension})",
                                    labels={'color': 'Cluster'},
                                    color_continuous_scale='Rainbow')
            st.plotly_chart(fig, use_container_width=True)
        
        # Optimize the comparative analysis section
        if st.checkbox("Model Performance and Clustering"):
            st.markdown("## Model Performance and Clustering")
            
            model_type = st.selectbox("Select model for clustering:", 
                                      ['KMeans', 'Agglomerative', 'GMM', 'Mean Shift', 'Birch', 'DBSCAN'])

            # Input parameters based on model type
            if model_type == 'KMeans':
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=8)
                model = KMeans(n_clusters=n_clusters)
            elif model_type == 'Agglomerative':
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=8)
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif model_type == 'GMM':
                n_components = st.slider("Number of Components", min_value=2, max_value=20, value=8)
                model = GaussianMixture(n_components=n_components)
            elif model_type == 'Mean Shift':
                bandwidth = st.slider("Bandwidth", min_value=0.1, max_value=10.0, value=2.0)
                model = MeanShift(bandwidth=bandwidth)
            elif model_type == 'Birch':
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=8)
                model = Birch(n_clusters=n_clusters)
            elif model_type == 'DBSCAN':
                eps = st.slider("Epsilon", min_value=0.1, max_value=10.0, value=0.5)
                min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            
            dim_reduction_technique = st.selectbox("Select Dimensionality Reduction Technique:", 
                                                   ['PCA', 'UMAP', 't-SNE'])
            dim_reduction_dimension = st.selectbox("Select Dimension:", ['2D', '3D'])
            
            result_key = f"{dim_reduction_technique}_{dim_reduction_dimension}"
            result = dim_reduction_results[result_key]
            
            predicted_labels = model.fit_predict(result)
            
            # Calculate clustering metrics
            silhouette_avg = silhouette_score(result, predicted_labels)
            davies_bouldin = davies_bouldin_score(result, predicted_labels)
            calinski_harabasz = calinski_harabasz_score(result, predicted_labels)
            trustworthiness_score = trustworthiness(scaled_features, result, n_neighbors=5)
            
            # Display metrics
            st.markdown("### Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Silhouette Score", f"{silhouette_avg:.2f}", help="Higher is better")
            col2.metric("Davies-Bouldin Index", f"{davies_bouldin:.2f}", help="Lower is better")
            col3.metric("Calinski-Harabasz Index", f"{calinski_harabasz:.2f}", help="Higher is better")
            col4.metric("Trustworthiness", f"{trustworthiness_score:.2f}", help="Higher is better")
            
            # Plot the clustering result
            if dim_reduction_dimension == '2D':
                fig = px.scatter(x=result[:, 0], y=result[:, 1], 
                                 color=predicted_labels, 
                                 title=f"{model_type} Clustering ({dim_reduction_technique} - {dim_reduction_dimension})",
                                 labels={'color': 'Cluster'},
                                 color_continuous_scale='Rainbow')
            else:
                fig = px.scatter_3d(x=result[:, 0], y=result[:, 1], z=result[:, 2],
                                    color=predicted_labels, 
                                    title=f"{model_type} Clustering ({dim_reduction_technique} - {dim_reduction_dimension})",
                                    labels={'color': 'Cluster'},
                                    color_continuous_scale='Rainbow')
            st.plotly_chart(fig, use_container_width=True)
        
        

if __name__ == "__main__":
    main()