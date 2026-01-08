import cv2
import time
import math
import requests
from ultralytics import YOLO

# --- Configuración de Parámetros ---
PXM_RATIO = 0.1  
PREDICTION_INTERVAL = 0.1 
API_URL = "https://dustin-unedible-bethany.ngrok-free.dev/api/posicion" # URL de tu API Flask
# -----------------------------------

def enviar_a_api(x, y, z):
    """Envía las coordenadas calculadas al servidor Flask."""
    payload = {
        "x": round(x, 2),
        "y": round(y, 2),
        "z": round(z, 2)
    }
    try:
        # Usamos un timeout pequeño para no congelar el flujo de video si la API tarda
        response = requests.post(API_URL, json=payload, timeout=0.05)
        if response.status_code == 200:
            print(f"🚀 API Update: X:{payload['x']} Y:{payload['y']} Z:{payload['z']}")
    except Exception as e:
        print(f"⚠️ Error de conexión con API: {e}")

def seleccionar_camara():
    camaras_disponibles = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: camaras_disponibles.append(i)
            cap.release()
    return min(camaras_disponibles) if camaras_disponibles else -1

def main():
    try:
        model = YOLO("my_model_2.pt")
        print("✅ Modelo cargado y sistema de visión listo.")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}"); return

    camera_index = seleccionar_camara()
    if camera_index == -1: return
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    last_prediction_time = time.time()
    annotated_frame = None 

    while True:
        ret, frame = cap.read()
        if not ret: break
        current_time = time.time()
        
        if (current_time - last_prediction_time) >= PREDICTION_INTERVAL:
            results = model(frame, verbose=False)
            
            ansuz_data = None
            cubo_morado_data = None
            max_conf_ansuz = -1.0
            max_conf_cubo = -1.0
            
            annotated_frame = frame.copy()

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    b = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, b)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)

                    # Seleccionar solo el Ansuz con mayor confianza
                    if class_name == "ansuz" and conf > max_conf_ansuz:
                        max_conf_ansuz = conf
                        ansuz_data = {'centro': (cx, cy), 'area': area, 'bbox': (x1, y1, x2, y2), 'conf': conf}

                    # Seleccionar solo el Cubo Morado con mayor confianza
                    elif class_name == "cubo_morado" and conf > max_conf_cubo:
                        max_conf_cubo = conf
                        cubo_morado_data = {'centro': (cx, cy), 'area': area, 'bbox': (x1, y1, x2, y2), 'conf': conf}

            # --- Procesamiento de Datos y Comunicación ---
            if ansuz_data and cubo_morado_data:
                # Cálculo de distancias relativas
                dx = (cubo_morado_data['centro'][0] - ansuz_data['centro'][0]) * PXM_RATIO
                dy = (ansuz_data['centro'][1] - cubo_morado_data['centro'][1]) * PXM_RATIO
                relacion_z = math.sqrt(ansuz_data['area'] / cubo_morado_data['area'])
                dz = (relacion_z - 1.0) * 10 
                
                # ENVÍO A LA API
                enviar_a_api(dx, dy, dz)

                # Visualización en pantalla
                cv2.line(annotated_frame, ansuz_data['centro'], cubo_morado_data['centro'], (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Rel X:{dx:.1f} Y:{dy:.1f} Z:{dz:.1f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Dibujar cuadros de detección
            for obj, color, label in [(ansuz_data, (0, 255, 255), "Ansuz"), (cubo_morado_data, (255, 0, 255), "Cubo")]:
                if obj:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} {obj['conf']:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            last_prediction_time = current_time
        
        display_frame = annotated_frame if annotated_frame is not None else frame
        cv2.imshow('Interfaz del gemelo fisico - ODIN-XR', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()