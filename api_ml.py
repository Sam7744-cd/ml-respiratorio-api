import os
import uuid
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from extract_features import extract_features

app = Flask(__name__)
CORS(app)

# RUTAS BASE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
AUDIOS_DIR = os.path.join(RECORDS_DIR, "audios")
LOGS_DIR = os.path.join(RECORDS_DIR, "logs")
PLOTS_DIR = os.path.join(RECORDS_DIR, "plots")

PENDING_DIR = os.path.join(AUDIOS_DIR, "pending")
ERROR_DIR = os.path.join(AUDIOS_DIR, "error")
HEALTHY_DIR = os.path.join(AUDIOS_DIR, "healthy")
ASTHMA_DIR = os.path.join(AUDIOS_DIR, "asthma")
BRONCHIAL_DIR = os.path.join(AUDIOS_DIR, "bronchial")

PLOTS_WAVEFORM_DIR = os.path.join(PLOTS_DIR, "waveforms")
PLOTS_SPECTROGRAM_DIR = os.path.join(PLOTS_DIR, "spectrograms")

PREDICTIONS_CSV = os.path.join(LOGS_DIR, "predicciones.csv")

# Crear carpetas si no existen
os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)
os.makedirs(HEALTHY_DIR, exist_ok=True)
os.makedirs(ASTHMA_DIR, exist_ok=True)
os.makedirs(BRONCHIAL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_WAVEFORM_DIR, exist_ok=True)
os.makedirs(PLOTS_SPECTROGRAM_DIR, exist_ok=True)

# CARGAR MODELO
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))

print(" Modelo ML cargado correctamente")

# GUARDAR LOG
def save_prediction_log(data):
    df_new = pd.DataFrame([data])

    if os.path.exists(PREDICTIONS_CSV):
        df_old = pd.read_csv(PREDICTIONS_CSV)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(PREDICTIONS_CSV, index=False)

# OBTENER CARPETA SEGÚN CLASE
def get_class_folder(pred_label):
    pred_label = pred_label.strip().lower()

    if pred_label == "healthy":
        return HEALTHY_DIR
    elif pred_label == "asthma":
        return ASTHMA_DIR
    elif pred_label == "bronchial":
        return BRONCHIAL_DIR
    else:
        return ERROR_DIR

# GENERAR VISUALIZACIONES
def generate_audio_plots(audio_path, base_name):
    y, sr = librosa.load(audio_path, sr=22050)

    waveform_filename = f"{base_name}_waveform.png"
    spectrogram_filename = f"{base_name}_spectrogram.png"

    waveform_path = os.path.join(PLOTS_WAVEFORM_DIR, waveform_filename)
    spectrogram_path = os.path.join(PLOTS_SPECTROGRAM_DIR, spectrogram_filename)

    # Forma de onda
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Forma de onda")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.tight_layout()
    plt.savefig(waveform_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Espectrograma Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Espectrograma Mel")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Mel)")
    plt.tight_layout()
    plt.savefig(spectrogram_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "waveform_filename": waveform_filename,
        "spectrogram_filename": spectrogram_filename,
    }

# SERVIR AUDIOS
@app.route("/audios/<path:filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIOS_DIR, filename)

# SERVIR WAVEFORMS
@app.route("/plots/waveforms/<path:filename>", methods=["GET"])
def serve_waveform(filename):
    return send_from_directory(PLOTS_WAVEFORM_DIR, filename)

# SERVIR ESPECTROGRAMAS
@app.route("/plots/spectrograms/<path:filename>", methods=["GET"])
def serve_spectrogram(filename):
    return send_from_directory(PLOTS_SPECTROGRAM_DIR, filename)
# PREDICCIÓN
@app.route("/predict-audio", methods=["POST"])
def predict_audio():
    pending_audio_path = None
    final_audio_path = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "El archivo está vacío"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        original_name = file.filename
        extension = os.path.splitext(original_name)[1].lower()

        if extension == "":
            extension = ".wav"

        filename = f"audio_{timestamp}_{unique_id}{extension}"
        base_name = os.path.splitext(filename)[0]

        # Guardar temporal
        pending_audio_path = os.path.join(PENDING_DIR, filename)
        file.save(pending_audio_path)

        # Extraer features
        features = extract_features(pending_audio_path)

        X = pd.DataFrame([features])
        non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric_cols:
            X = X.drop(columns=non_numeric_cols)

        # Escalar y predecir
        X_scaled = scaler.transform(X)

        pred_encoded = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]

        pred_label = encoder.inverse_transform([pred_encoded])[0].strip().lower()
        confidence = float(probs[pred_encoded])

        # Mover audio a carpeta de clase
        class_folder = get_class_folder(pred_label)
        final_audio_path = os.path.join(class_folder, filename)
        shutil.move(pending_audio_path, final_audio_path)

        relative_audio_path = os.path.relpath(final_audio_path, AUDIOS_DIR).replace("\\", "/")

        # Generar gráficas
        plots = generate_audio_plots(final_audio_path, base_name)

        # URL base dinámica para local o Render
        base_url = request.host_url.rstrip("/")

        waveform_url = f"{base_url}/plots/waveforms/{plots['waveform_filename']}"
        spectrogram_url = f"{base_url}/plots/spectrograms/{plots['spectrogram_filename']}"
        audio_url = f"{base_url}/audios/{relative_audio_path}"

        # Features resumidas
        features_resumen = {
            "duration": round(float(features.get("duration", 0)), 4),
            "rms_mean": round(float(features.get("rms_mean", 0)), 6),
            "zcr_mean": round(float(features.get("zcr_mean", 0)), 6),
            "centroid_mean": round(float(features.get("centroid_mean", 0)), 4),
            "bandwidth_mean": round(float(features.get("bandwidth_mean", 0)), 4),
        }

        # Guardar log
        log_data = {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "archivo": filename,
            "clase_guardada": pred_label,
            "ruta_audio": f"records/audios/{relative_audio_path}",
            "prediccion": pred_label,
            "confianza": round(confidence, 4),
            "waveform_url": waveform_url,
            "spectrogram_url": spectrogram_url,
        }

        save_prediction_log(log_data)

        return jsonify({
            "mensaje": "Predicción realizada correctamente",
            "archivo_guardado": filename,
            "clase_guardada": pred_label,
            "ruta_audio": f"records/audios/{relative_audio_path}",
            "audio_url": audio_url,
            "waveform_url": waveform_url,
            "spectrogram_url": spectrogram_url,
            "features_resumen": features_resumen,
            "prediccion": pred_label,
            "confianza": round(confidence, 4)
        })

    except Exception as e:
        print(" ERROR EN ML:", str(e))

        if pending_audio_path and os.path.exists(pending_audio_path):
            error_path = os.path.join(ERROR_DIR, os.path.basename(pending_audio_path))
            try:
                shutil.move(pending_audio_path, error_path)
            except Exception:
                pass

        return jsonify({"error": str(e)}), 500
# RUN
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)