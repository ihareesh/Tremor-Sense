import socket
import pickle
import joblib
import numpy as np
import pandas as pd
import librosa
import time
import warnings
from collections import deque
from scipy.fft import fft, fftfreq
from datetime import datetime
import os

warnings.filterwarnings("ignore")

# ================= PATHS =================
VOWEL_MODEL_PATH = r"F:\Parkinson_disease\vowel_model.pkl"
TREMOR_MODEL_PATH = r"F:\Parkinson_disease\7.Tremor_Models\Model_3\tremor_model.pkl"

BASE_DIR = r"F:\Parkinson_disease\7.Tremor_Models\Model_3"
LOG_DIR = os.path.join(BASE_DIR, "integrated_voice_imu_logs")
os.makedirs(LOG_DIR, exist_ok=True)

session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"integrated_session_{session_start}.csv")

# ================= UDP =================
UDP_IP = "0.0.0.0"
AUDIO_PORT = 5005
IMU_PORT = 1234
PWM_CMD_PORT = 1235
ESP32_IP = "192.168.4.1"

PACKET_SAMPLES = 512
SAMPLE_RATE = 16000

# ================= VOICE SETTINGS =================
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.25

VOICE_MARGIN_THRESHOLD = 0.18
VOICE_REQUIRED_REPEAT = 3
WAKE_REQUIRED_REPEAT = 4

MIN_AUDIO_RMS = 2500
MAX_AUDIO_RMS = 26000

COMMAND_MAP = {
    "O": "WAKE",
    "A": "DRINK",
    "E": "EAT",
    "I": "PILL",
    "U": "STOP"
}

PER_LABEL_THRESHOLD = {
    "O": 0.55,
    "A": 0.46,
    "E": 0.48,
    "I": 0.42,
    "U": 0.38
}

# ================= IMU SETTINGS =================
WINDOW_SIZE = 192
STEP_SIZE = 20

CONF_THRESHOLD = 0.70
FREQ_LOW = 3.0
FREQ_HIGH = 7.0
POWER_THRESHOLD = 5.0
MIN_SEND_INTERVAL = 0.25

PWM_MAX = 160

MODE_BASELINE_PWM = {
    "DRINK": 60,
    "EAT": 20,
    "PILL": 20,
    "IDLE": 0
}

MODE_MAX_PWM = {
    "DRINK": 160,
    "EAT": 160,
    "PILL": 100,
    "IDLE": 0
}

POWER_MIN = 5
POWER_MAX = 150

MAX_TARGET_CHANGE_PER_STEP = 25
MAX_SMOOTHED_CHANGE_PER_STEP = 15

# ================= LOAD MODELS =================
print("Loading vowel model...")
with open(VOWEL_MODEL_PATH, "rb") as f:
    vowel_data = pickle.load(f)

vowel_model = vowel_data["model"]
vowel_scaler = vowel_data["scaler"]
vowel_le = vowel_data["label_encoder"]

print("Vowel model loaded:", [str(x) for x in vowel_le.classes_])

print("Loading tremor model...")
tremor_model = joblib.load(TREMOR_MODEL_PATH)
print("Tremor model loaded successfully")

# ================= SOCKETS =================
audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
audio_sock.bind((UDP_IP, AUDIO_PORT))
audio_sock.settimeout(1.0)

imu_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
imu_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
imu_sock.bind((UDP_IP, IMU_PORT))
imu_sock.settimeout(0.01)

pwm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_command_udp(cmd):
    pwm_sock.sendto(cmd.encode(), (ESP32_IP, PWM_CMD_PORT))
    print(f"UDP SENT TO ESP32: {cmd}")


# ================= STATE VARIABLES =================
system_awake = False
current_mode = "IDLE"

last_sent_pwm = 0
last_send_time = time.time()
last_target_pwm = 0
last_smoothed_pwm = 0

pwm_buffer = deque(maxlen=5)

imu_buffer = deque(maxlen=WINDOW_SIZE)
imu_time_buffer = deque(maxlen=WINDOW_SIZE)

sample_counter = 0
last_prediction_sample = 0
prediction_count = 0

audio_buffer = []

last_voice_command = None
last_voice_time = 0

candidate_label = None
candidate_count = 0

session_log = []


# ================= HELPERS =================
def clamp(value, low, high):
    return max(low, min(value, high))


def get_mode_baseline_pwm(mode):
    return MODE_BASELINE_PWM.get(mode, 0)


def get_mode_max_pwm(mode):
    return MODE_MAX_PWM.get(mode, 0)


def classify_severity(power):
    if power < 75:
        return "MILD"
    elif power < 90:
        return "MODERATE"
    return "SEVERE"


def map_power_to_pwm(power, mode):
    baseline = get_mode_baseline_pwm(mode)
    max_pwm = get_mode_max_pwm(mode)

    if mode == "IDLE":
        return 0

    if power < POWER_THRESHOLD:
        return baseline

    power_clamped = clamp(power, POWER_MIN, POWER_MAX)
    ratio = (power_clamped - POWER_MIN) / (POWER_MAX - POWER_MIN)
    pwm = baseline + ratio * (max_pwm - baseline)

    return int(clamp(round(pwm), baseline, max_pwm))


def rate_limit_target_pwm(target_pwm):
    global last_target_pwm

    if target_pwm > last_target_pwm + MAX_TARGET_CHANGE_PER_STEP:
        target_pwm = last_target_pwm + MAX_TARGET_CHANGE_PER_STEP
    elif target_pwm < last_target_pwm - MAX_TARGET_CHANGE_PER_STEP:
        target_pwm = last_target_pwm - MAX_TARGET_CHANGE_PER_STEP

    last_target_pwm = int(target_pwm)
    return int(target_pwm)


def smooth_pwm(target_pwm, mode):
    global last_smoothed_pwm

    if mode == "IDLE":
        pwm_buffer.clear()
        last_smoothed_pwm = 0
        return 0

    pwm_buffer.append(target_pwm)
    avg_pwm = int(round(sum(pwm_buffer) / len(pwm_buffer)))

    if avg_pwm > last_smoothed_pwm + MAX_SMOOTHED_CHANGE_PER_STEP:
        avg_pwm = last_smoothed_pwm + MAX_SMOOTHED_CHANGE_PER_STEP
    elif avg_pwm < last_smoothed_pwm - MAX_SMOOTHED_CHANGE_PER_STEP:
        avg_pwm = last_smoothed_pwm - MAX_SMOOTHED_CHANGE_PER_STEP

    baseline = get_mode_baseline_pwm(mode)
    avg_pwm = clamp(avg_pwm, baseline, get_mode_max_pwm(mode))

    last_smoothed_pwm = int(avg_pwm)
    return int(avg_pwm)


def send_pwm(pwm, force=False):
    global last_sent_pwm, last_send_time

    now = time.time()
    pwm = int(clamp(pwm, 0, PWM_MAX))

    if not force:
        if now - last_send_time < MIN_SEND_INTERVAL:
            return False
        if abs(pwm - last_sent_pwm) < 3:
            return False

    msg = "STOP" if pwm <= 0 else f"PWM:{pwm}"
    pwm_sock.sendto(msg.encode(), (ESP32_IP, PWM_CMD_PORT))

    print(f"PWM SENT: {msg}")

    last_sent_pwm = pwm
    last_send_time = now
    return True


def activate_mode(mode):
    global current_mode, last_target_pwm, last_smoothed_pwm

    if current_mode == mode:
        return

    current_mode = mode
    baseline = get_mode_baseline_pwm(mode)

    pwm_buffer.clear()
    last_target_pwm = baseline
    last_smoothed_pwm = baseline

    send_pwm(baseline, force=True)
    print(f"{mode} MODE selected | baseline PWM = {baseline}")


def stop_system():
    global system_awake, current_mode, last_target_pwm, last_smoothed_pwm, last_sent_pwm

    system_awake = False
    current_mode = "IDLE"

    pwm_buffer.clear()
    last_target_pwm = 0
    last_smoothed_pwm = 0
    last_sent_pwm = 0

    send_pwm(0, force=True)
    print("SYSTEM STOPPED | Motor OFF | Only WAKE accepted")


# ================= VOICE FUNCTIONS =================
def audio_rms(audio):
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)
    return np.sqrt(np.mean(audio ** 2))


def preprocess_audio(audio):
    audio = audio.astype(np.float32)
    audio = audio - np.mean(audio)

    max_val = np.max(np.abs(audio))
    if max_val < 50:
        return None

    audio = audio / max_val

    total_len = len(audio)
    start = int(total_len * 0.20)
    end = int(total_len * 0.90)
    audio = audio[start:end]

    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE]

    return audio.astype(np.float32)


def extract_vowel_features(audio, sr=SAMPLE_RATE):
    audio_norm = audio.astype(np.float32)

    max_val = np.max(np.abs(audio_norm))
    if max_val <= 0:
        return None

    audio_norm = audio_norm / max_val
    features = []

    mfcc = librosa.feature.mfcc(y=audio_norm, sr=sr, n_mfcc=20, n_mels=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.mean(mfcc_delta2, axis=1))

    fft_vals = np.abs(np.fft.rfft(audio_norm))
    freqs = np.fft.rfftfreq(len(audio_norm), 1 / sr)

    def band_energy(low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.sum(fft_vals[mask])

    total = band_energy(100, 8000) + 1e-10

    bands = [
        (200, 350), (350, 450), (450, 550), (550, 650),
        (650, 800), (800, 1000), (1000, 1400), (1400, 1800),
        (1800, 2100), (2100, 2400), (2400, 2700),
        (2700, 3500), (3500, 5000)
    ]

    for low, high in bands:
        features.append(band_energy(low, high) / total)

    f1_e = band_energy(450, 650)
    f1_i = band_energy(350, 450)
    f2_e = band_energy(1800, 2100)
    f2_i = band_energy(2100, 2400)

    features.append(f1_e / (f1_i + 1e-10))
    features.append(f2_e / (f2_i + 1e-10))
    features.append((f1_e + f2_e) / (f1_i + f2_i + 1e-10))

    spec_centroid = librosa.feature.spectral_centroid(y=audio_norm, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_norm, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio_norm, sr=sr)
    spec_flatness = librosa.feature.spectral_flatness(y=audio_norm)
    spec_contrast = librosa.feature.spectral_contrast(y=audio_norm, sr=sr)

    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))
    features.append(np.mean(spec_bandwidth))
    features.append(np.mean(spec_rolloff))
    features.append(np.mean(spec_flatness))
    features.extend(np.mean(spec_contrast, axis=1))

    chroma = librosa.feature.chroma_stft(y=audio_norm, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    zcr = librosa.feature.zero_crossing_rate(audio_norm)
    rms = librosa.feature.rms(y=audio_norm)

    features.append(np.mean(zcr))
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


def apply_vowel_correction(proba):
    labels = [str(x) for x in vowel_le.classes_]
    probs = dict(zip(labels, proba))

    for key in ["A", "E", "I", "O", "U", "noise"]:
        probs.setdefault(key, 0.0)

    A = probs["A"]
    E = probs["E"]
    I = probs["I"]
    O = probs["O"]
    U = probs["U"]
    N = probs["noise"]

    if N >= 0.35:
        arr = np.array([A, E, I, O, U, N], dtype=np.float32)
        arr = arr / np.sum(arr)
        return "noise", arr[5], arr, 0.0

    if A >= 0.24 and O < 0.45 and U < 0.40:
        A += 0.18

    if I > E and (I - E) < 0.35:
        E += 0.20

    if O > U and U >= 0.22 and (O - U) <= 0.22:
        U += 0.30
        O -= 0.12

    if not system_awake:
        if O >= 0.45 and U < 0.35:
            O += 0.12
        elif O < 0.45:
            O *= 0.55
    else:
        O = 0.0

    if U >= 0.42:
        U += 0.18

    corrected = {
        "A": A,
        "E": E,
        "I": I,
        "O": O,
        "U": U,
        "noise": N
    }

    ordered = ["A", "E", "I", "O", "U", "noise"]
    values = np.array([corrected[x] for x in ordered], dtype=np.float32)
    values = np.clip(values, 0, None)

    if np.sum(values) == 0:
        return None, 0.0, None, 0.0

    corrected_proba = values / np.sum(values)
    final_idx = np.argmax(corrected_proba)

    final_label = ordered[final_idx]
    final_conf = corrected_proba[final_idx]

    sorted_probs = np.sort(corrected_proba)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    return final_label, final_conf, corrected_proba, margin


def predict_vowel(raw_audio):
    rms_now = audio_rms(raw_audio)

    if rms_now < MIN_AUDIO_RMS:
        return None, 0.0, None, 0.0

    if rms_now > MAX_AUDIO_RMS:
        return None, 0.0, None, 0.0

    audio = preprocess_audio(raw_audio)
    if audio is None:
        return None, 0.0, None, 0.0

    all_probs = []
    shifts = [0, 300, 600, 900]

    for shift in shifts:
        segment = audio[shift:shift + SAMPLE_RATE]

        if len(segment) < SAMPLE_RATE:
            segment = np.pad(segment, (0, SAMPLE_RATE - len(segment)))

        features = extract_vowel_features(segment)
        if features is None:
            continue

        features_scaled = vowel_scaler.transform(features.reshape(1, -1))
        proba = vowel_model.predict_proba(features_scaled)[0]
        all_probs.append(proba)

    if len(all_probs) == 0:
        return None, 0.0, None, 0.0

    avg_proba = np.mean(all_probs, axis=0)
    return apply_vowel_correction(avg_proba)


def is_voice_allowed(label):
    if not system_awake:
        return label == "O"

    return label in ["A", "E", "I", "U"]


def process_voice_command(label, confidence, margin):
    global system_awake, current_mode, last_voice_command, last_voice_time
    global candidate_label, candidate_count

    command = COMMAND_MAP.get(label)
    now = time.time()

    if command is None:
        return

    if command == last_voice_command:
        return

    if system_awake:
        if command == "DRINK" and current_mode == "DRINK":
            return
        if command == "EAT" and current_mode == "EAT":
            return
        if command == "PILL" and current_mode == "PILL":
            return

    print("=" * 60)
    print(f"VOICE DETECTED : {label}")
    print(f"COMMAND        : {command}")
    print(f"CONFIDENCE     : {confidence * 100:.1f}%")
    print(f"MARGIN         : {margin * 100:.1f}%")
    print("=" * 60)

    send_command_udp(label)

    if label == "O":
        system_awake = True
        current_mode = "IDLE"
        pwm_buffer.clear()
        send_pwm(0, force=True)
        print("SYSTEM AWAKE | Select A / E / I")

    elif label == "A" and system_awake:
        activate_mode("DRINK")

    elif label == "E" and system_awake:
        activate_mode("EAT")

    elif label == "I" and system_awake:
        activate_mode("PILL")

    elif label == "U":
        stop_system()

    last_voice_command = command
    last_voice_time = now
    candidate_label = None
    candidate_count = 0


# ================= IMU FUNCTIONS =================
def extract_imu_features(df, fs):
    features = {}
    sensor_cols = ["ax", "ay", "az", "gx", "gy", "gz"]

    for col in sensor_cols:
        sig = df[col]
        features[f"{col}_mean"] = sig.mean()
        features[f"{col}_std"] = sig.std()
        features[f"{col}_var"] = sig.var()
        features[f"{col}_min"] = sig.min()
        features[f"{col}_max"] = sig.max()
        features[f"{col}_range"] = sig.max() - sig.min()
        features[f"{col}_rms"] = np.sqrt(np.mean(sig ** 2))

    acc_mag = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)

    features["acc_mag_mean"] = acc_mag.mean()
    features["acc_mag_std"] = acc_mag.std()
    features["acc_mag_var"] = acc_mag.var()
    features["acc_mag_min"] = acc_mag.min()
    features["acc_mag_max"] = acc_mag.max()
    features["acc_mag_range"] = acc_mag.max() - acc_mag.min()
    features["acc_mag_rms"] = np.sqrt(np.mean(acc_mag ** 2))

    yf = np.abs(fft(acc_mag))
    xf = fftfreq(len(acc_mag), 1 / fs)

    positive = xf > 0
    xf = xf[positive]
    yf = yf[positive]

    tremor_mask = (xf >= FREQ_LOW) & (xf <= FREQ_HIGH)

    if np.any(tremor_mask):
        tremor_xf = xf[tremor_mask]
        tremor_yf = yf[tremor_mask]
        tremor_band_freq = tremor_xf[np.argmax(tremor_yf)]
        tremor_power = np.sum(tremor_yf)
    else:
        tremor_band_freq = 0.0
        tremor_power = 0.0

    features["dominant_freq"] = tremor_band_freq
    features["tremor_band_power"] = tremor_power
    features["energy"] = np.sum(acc_mag ** 2)
    features["SMA"] = np.abs(df[["ax", "ay", "az"]]).sum().sum() / len(df)

    return pd.DataFrame([features]), tremor_band_freq, tremor_power


# ================= MAIN =================
def main():
    global audio_buffer, sample_counter, last_prediction_sample, prediction_count
    global candidate_label, candidate_count

    print("\nWaiting for ESP32 audio + IMU data...")
    print("Initial state: STOPPED. Only strong repeated O / WAKE accepted.")
    print("After wake: O ignored. A/E/I/U accepted.")
    print("O=Wake | A=Drink | E=Eat | I=Pill | U=Stop\n")

    while True:
        try:
            data, addr = audio_sock.recvfrom(PACKET_SAMPLES * 2 + 100)
            print(f"Audio connected from {addr[0]}")
            break
        except socket.timeout:
            print("Waiting for audio packets...")

    audio_sock.settimeout(0.001)

    try:
        while True:
            try:
                while True:
                    data, _ = audio_sock.recvfrom(PACKET_SAMPLES * 2 + 100)
                    samples = np.frombuffer(data, dtype=np.int16)
                    audio_buffer.extend(samples)
            except socket.timeout:
                pass

            window_size_audio = int(SAMPLE_RATE * WINDOW_SECONDS)
            step_size_audio = int(SAMPLE_RATE * STEP_SECONDS)

            if len(audio_buffer) >= window_size_audio:
                current_audio = np.array(audio_buffer[-window_size_audio:], dtype=np.int16)
                label, confidence, proba, margin = predict_vowel(current_audio)

                if label is not None and label != "noise" and is_voice_allowed(label):
                    required_conf = PER_LABEL_THRESHOLD.get(label, 0.45)

                    if confidence >= required_conf and margin >= VOICE_MARGIN_THRESHOLD:
                        if label == candidate_label:
                            candidate_count += 1
                        else:
                            candidate_label = label
                            candidate_count = 1

                        required_repeat = WAKE_REQUIRED_REPEAT if (not system_awake and label == "O") else VOICE_REQUIRED_REPEAT

                        if candidate_count >= required_repeat:
                            process_voice_command(label, confidence, margin)
                    else:
                        candidate_label = None
                        candidate_count = 0
                else:
                    candidate_label = None
                    candidate_count = 0

                audio_buffer = audio_buffer[step_size_audio:]

            try:
                while True:
                    imu_data, _ = imu_sock.recvfrom(1024)
                    vals = imu_data.decode().strip().split(",")

                    if len(vals) < 7:
                        continue

                    try:
                        t = float(vals[0])
                        ax, ay, az, gx, gy, gz = map(float, vals[1:7])
                    except ValueError:
                        continue

                    imu_buffer.append([ax, ay, az, gx, gy, gz])
                    imu_time_buffer.append(t)
                    sample_counter += 1

            except socket.timeout:
                pass

            if len(imu_buffer) < WINDOW_SIZE:
                continue

            if sample_counter - last_prediction_sample < STEP_SIZE:
                continue

            last_prediction_sample = sample_counter
            prediction_count += 1

            df = pd.DataFrame(imu_buffer, columns=["ax", "ay", "az", "gx", "gy", "gz"])

            duration = imu_time_buffer[-1] - imu_time_buffer[0]
            fs = len(imu_time_buffer) / duration if duration > 0 else 75.0

            features, freq, power = extract_imu_features(df, fs)

            acc_mag = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
            rms_val = np.sqrt(np.mean(acc_mag ** 2))

            pred = tremor_model.predict(features)[0]
            conf = np.max(tremor_model.predict_proba(features))

            state = "NORMAL"
            severity = "-"
            raw_target_pwm = 0
            target_pwm = 0
            smoothed_pwm = 0
            sent_now = False

            if not system_awake or current_mode == "IDLE":
                state = "IDLE"
                send_pwm(0)

            else:
                baseline = get_mode_baseline_pwm(current_mode)

                valid_tremor = (
                    pred == 1
                    and conf >= CONF_THRESHOLD
                    and FREQ_LOW <= freq <= FREQ_HIGH
                    and power >= POWER_THRESHOLD
                )

                if valid_tremor:
                    state = "TREMOR"
                    severity = classify_severity(power)
                    raw_target_pwm = map_power_to_pwm(power, current_mode)
                else:
                    state = "NORMAL"
                    raw_target_pwm = baseline

                target_pwm = rate_limit_target_pwm(raw_target_pwm)
                smoothed_pwm = smooth_pwm(target_pwm, current_mode)
                sent_now = send_pwm(smoothed_pwm)

            session_log.append({
                "prediction_no": prediction_count,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "awake": system_awake,
                "mode": current_mode,
                "state": state,
                "confidence": float(conf),
                "tremor_frequency_hz": float(freq),
                "power": float(power),
                "rms": float(rms_val),
                "severity": severity,
                "raw_target_pwm": int(raw_target_pwm),
                "target_pwm": int(target_pwm),
                "smoothed_pwm": int(smoothed_pwm),
                "sent_pwm": int(last_sent_pwm),
                "command_sent_now": sent_now
            })

            if prediction_count % 10 == 0:
                print(
                    f"[{prediction_count:04d}] "
                    f"AWAKE={system_awake} MODE={current_mode} STATE={state} "
                    f"CONF={conf:.2f} FREQ={freq:.2f}Hz POWER={power:.2f} "
                    f"RMS={rms_val:.3f} PWM={last_sent_pwm}"
                )

            if prediction_count % 20 == 0:
                pd.DataFrame(session_log).to_csv(log_path, index=False)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        send_pwm(0, force=True)

        if len(session_log) > 0:
            pd.DataFrame(session_log).to_csv(log_path, index=False)
            print("Session log saved:", log_path)

        audio_sock.close()
        imu_sock.close()
        pwm_sock.close()
        print("Closed all UDP connections")

main() 
