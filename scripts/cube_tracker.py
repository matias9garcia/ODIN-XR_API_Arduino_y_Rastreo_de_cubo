import cv2
import cv2.aruco as aruco
import numpy as np
import time
import requests
import math
import threading
import queue

# --- Selección de cámara USB ---
def seleccionar_camara():
    print("🔍 Buscando cámaras disponibles...")
    camaras_disponibles = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Cámara detectada en índice {i}")
                camaras_disponibles.append(i)
        cap.release()

    if not camaras_disponibles:
        print("⚠️ No se detectó ninguna cámara. Verifica la conexión USB.")
        return 0
    
    camara_seleccionada = min(camaras_disponibles)
    print(f"🎥 Usando cámara índice {camara_seleccionada}")
    return camara_seleccionada

# --- Inicializar cámara ---
cam_index = seleccionar_camara()
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

# --- Parámetros cámara ---
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# --- Parámetros ArUco ---
marker_length = 0.05
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# --- IDs ---
ID_REF = 0
ID_CUBO = [1,2,3,4,5,6]

# --- Offset Unity (ADAPTADO AL CENTRO DEL OBJETO) ---
offset_unity = np.array([-0.4, 1.05, 0.8])

# --- API destino y Hilos ---
API_URL = "http://10.108.234.112:5000/api/posicion"
data_queue = queue.Queue(maxsize=1) 

def sender_thread():
    """Hilo dedicado a enviar datos sin congelar la cámara"""
    while True:
        payload = data_queue.get()
        try:
            requests.post(API_URL, json=payload, timeout=1.0) 
        except requests.exceptions.RequestException:
            pass 
        data_queue.task_done()

t = threading.Thread(target=sender_thread)
t.daemon = True
t.start()

# --- Variables referencia ---
ref_rvec = None
ref_tvec = None

# --- Frecuencia mínima envío ---
MIN_INTERVAL = 0.1
last_send = 0

# --- Filtro anti-spike ---
ultima_pos = None             
pos_anterior_bruta = None      
SPIKE_THRESHOLD = 0.7         

def es_spike(pos_actual, ultima_pos, pos_anterior_bruta):
    if ultima_pos is None or pos_anterior_bruta is None:
        return False  

    dist_actual_ultima = np.linalg.norm(pos_actual - ultima_pos)
    dist_bruta_ultima = np.linalg.norm(pos_anterior_bruta - ultima_pos)

    if dist_actual_ultima > SPIKE_THRESHOLD and dist_bruta_ultima < SPIKE_THRESHOLD:
        return True
    return False

def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2,1], R[2,2])
        yaw = math.atan2(-R[2,0], sy)
        roll = math.atan2(R[1,0], R[0,0])
    else:
        pitch = math.atan2(-R[1,2], R[1,1])
        yaw = math.atan2(-R[2,0], sy)
        roll = 0
    return np.degrees([pitch, yaw, roll])

def transformar_a_referencia_global(rvec_obj, tvec_obj, rvec_ref, tvec_ref):
    R_ref, _ = cv2.Rodrigues(rvec_ref)
    R_obj, _ = cv2.Rodrigues(rvec_obj)
    R_global = R_ref.T @ R_obj
    rvec_global, _ = cv2.Rodrigues(R_global)
    t_global = R_ref.T @ (tvec_obj.reshape(3,1) - tvec_ref.reshape(3,1))
    return rvec_global, t_global


# -------------------- Bucle principal --------------------
print("🚀 Iniciando detección con Referencia Unity fijada. Presiona 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        ids_flat = ids.flatten()
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        # Dibujar ejes visuales
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

        if ID_REF in ids_flat:
            idx_ref = np.where(ids_flat == ID_REF)[0][0]
            ref_rvec = rvecs[idx_ref]
            ref_tvec = tvecs[idx_ref]

            for id_cara in ID_CUBO:
                if id_cara in ids_flat:
                    idx_cara = np.where(ids_flat == id_cara)[0][0]
                    rvec_cara = rvecs[idx_cara]
                    tvec_cara = tvecs[idx_cara]

                    # 1. Obtenemos posición relativa al marcador (0,0,0 si están juntos)
                    rvec_g, tvec_g = transformar_a_referencia_global(rvec_cara, tvec_cara, ref_rvec, ref_tvec)
                    pitch, yaw, roll = rvec_to_euler(rvec_g)

                    x_cv, y_cv, z_cv = tvec_g.flatten()
                    
                    # 2. Convertimos ejes y SUMAMOS LA POSICIÓN DEL OBJETO EN UNITY
                    # Esto hace que la referencia Aruco = Centro Objeto Unity
                    pos_unity = np.array([-x_cv, -z_cv, -y_cv]) + offset_unity
                    x, y, z = pos_unity

                    pos_actual = np.array([x, y, z])
                    
                    # Filtro anti-spike
                    spike = es_spike(pos_actual, ultima_pos, pos_anterior_bruta)
                    if spike:
                        print(f"⛔ Spike en ID {id_cara}")
                        pos_anterior_bruta = pos_actual
                        continue

                    pos_anterior_bruta = pos_actual

                    if time.time() - last_send >= MIN_INTERVAL:
                        payload = {
                            "timestamp": int(time.time() * 1000),
                            "x": float(x), "y": float(y), "z": float(z),
                            "pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)
                        }
                        
                        if not data_queue.full():
                            data_queue.put(payload)
                            ultima_pos = pos_actual 
                        
                        last_send = time.time()

                    print(f"ID {id_cara} -> Unity: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    cv2.imshow("Cubo ArUco (Ejes Visibles)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()