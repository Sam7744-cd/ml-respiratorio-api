import numpy as np
import librosa

def extract_features(file_path):
    """
    Extrae características acústicas de un archivo WAV
    para clasificación de sonidos respiratorios.
    """
    try:
        # CARGA DEL AUDIO
        y, sr = librosa.load(file_path, sr=22050)

        if y is None or len(y) < 100:
            raise ValueError("Audio vacío o demasiado corto")

        # PREPROCESAMIENTO
        # Normalización de amplitud
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # Eliminación de silencios al inicio y final
        y, _ = librosa.effects.trim(y)

        if y is None or len(y) < 100:
            raise ValueError("Audio demasiado corto después de eliminar silencios")

        # FEATURES BÁSICAS
        features = {}

        features["duration"] = float(librosa.get_duration(y=y, sr=sr))

        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["centroid_mean"] = float(np.mean(centroid))
        features["centroid_std"] = float(np.std(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["bandwidth_mean"] = float(np.mean(bandwidth))
        features["bandwidth_std"] = float(np.std(bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["rolloff_mean"] = float(np.mean(rolloff))
        features["rolloff_std"] = float(np.std(rolloff))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        for i, val in enumerate(mfcc_mean):
            features[f"mfcc_mean_{i+1}"] = float(val)

        for i, val in enumerate(mfcc_std):
            features[f"mfcc_std_{i+1}"] = float(val)

        # ENERGÍA POR BANDAS
        stft = np.abs(librosa.stft(y))

        low_band = stft[0:50, :]
        mid_band = stft[50:150, :]
        high_band = stft[150:300, :]

        features["low_energy"] = float(np.mean(low_band))
        features["mid_energy"] = float(np.mean(mid_band))
        features["high_energy"] = float(np.mean(high_band))

        # CHROMA Y CONTRAST
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["contrast_mean"] = float(np.mean(contrast))
        features["contrast_std"] = float(np.std(contrast))

        return features

    except Exception as e:
        raise ValueError(f"Error extrayendo características de {file_path}: {e}")