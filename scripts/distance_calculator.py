import cv2
import time
import math
from ultralytics import YOLO

# --- Configuración de Parámetros ---
PXM_RATIO = 0.1  
PREDICTION_INTERVAL = 0.1 
# -----------------------------------

def seleccionar_camara():
    camaras_disponibles = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: camaras_disponibles.append(i)
            cap.release()
    return max(camaras_disponibles) if camaras_disponibles else -1

def main():
    try:
        model = YOLO("my_model_2.pt")
        print("✅ Modelo cargado y listo.")
    except Exception as e:
        print(f"❌ Error: {e}"); return

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
            
            # Inicializamos variables para guardar el MEJOR objeto detectado
            ansuz_data = None
            cubo_morado_data = None
            max_conf_ansuz = -1.0
            max_conf_cubo = -1.0
            
            annotated_frame = frame.copy()

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0]) # Nivel de confianza (0.0 a 1.0)
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    b = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, b)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)

                    # --- FILTRO PARA EL MEJOR ANSUZ ---
                    if class_name == "ansuz":
                        if conf > max_conf_ansuz:
                            max_conf_ansuz = conf
                            ansuz_data = {
                                'centro': (cx, cy), 
                                'area': area, 
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf
                            }

                    # --- FILTRO PARA EL MEJOR CUBO MORADO ---
                    elif class_name == "cubo_morado":
                        if conf > max_conf_cubo:
                            max_conf_cubo = conf
                            cubo_morado_data = {
                                'centro': (cx, cy), 
                                'area': area, 
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf
                            }

            # --- DIBUJO Y CÁLCULOS (Solo para los mejores candidatos) ---
            if ansuz_data:
                x1, y1, x2, y2 = ansuz_data['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Ansuz {ansuz_data['conf']:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if cubo_morado_data:
                x1, y1, x2, y2 = cubo_morado_data['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(annotated_frame, f"Cubo {cubo_morado_data['conf']:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            if ansuz_data and cubo_morado_data:
                dx = (cubo_morado_data['centro'][0] - ansuz_data['centro'][0]) * PXM_RATIO
                dy = (ansuz_data['centro'][1] - cubo_morado_data['centro'][1]) * PXM_RATIO
                relacion_z = math.sqrt(ansuz_data['area'] / cubo_morado_data['area'])
                dz = (relacion_z - 1.0) * 10 
                distancia_3d = math.sqrt(dx**2 + dy**2 + dz**2)

                cv2.line(annotated_frame, ansuz_data['centro'], cubo_morado_data['centro'], (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Dist: {distancia_3d:.1f}cm", 
                            (cubo_morado_data['centro'][0], cubo_morado_data['centro'][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            last_prediction_time = current_time
        
        display_frame = annotated_frame if annotated_frame is not None else frame
        cv2.imshow('Módulo de visión artificial - ODIN-XR', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()