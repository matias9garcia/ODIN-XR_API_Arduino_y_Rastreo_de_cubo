import cv2
import time
import math
from ultralytics import YOLO

# --- Configuración de Parámetros ---
PXM_RATIO = 0.1  
PREDICTION_INTERVAL = 0.1 # Reducido a 0.1 para que sea más reactivo
# -----------------------------------

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
        model = YOLO("my_model.pt")
        print("✅ Modelo cargado y listo.")
    except Exception as e:
        print(f"❌ Error: {e}"); return

    camera_index = seleccionar_camara()
    if camera_index == -1: return
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    last_prediction_time = time.time()
    # Inicializamos annotated_frame con un frame vacío para evitar errores al inicio
    annotated_frame = None 

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        
        # Realizar detección cada cierto intervalo
        if (current_time - last_prediction_time) >= PREDICTION_INTERVAL:
            results = model(frame, verbose=False) # verbose=False limpia la consola
            cubos_data = [] 
            
            # Siempre empezamos con una copia limpia del frame actual
            annotated_frame = frame.copy()

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    b = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, b)

                    # --- DIBUJO PARA EL BRACCIO ---
                    if class_name == "Braccio":
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(annotated_frame, "Braccio", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # --- DIBUJO Y LÓGICA PARA CUBOS ---
                    elif class_name == "cubo_morado":
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        cubos_data.append({'centro': (cx, cy), 'area': area})
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(annotated_frame, "Cubo Morado", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # --- CÁLCULOS 3D SI HAY 2 CUBOS ---
            if len(cubos_data) >= 2:
                c1, c2 = cubos_data[0], cubos_data[1]
                dx = (c2['centro'][0] - c1['centro'][0]) * PXM_RATIO
                dy = (c1['centro'][1] - c2['centro'][1]) * PXM_RATIO
                relacion_z = math.sqrt(c1['area'] / c2['area'])
                dz = (relacion_z - 1.0) * 10 
                distancia_3d = math.sqrt(dx**2 + dy**2 + dz**2)

                cv2.line(annotated_frame, c1['centro'], c2['centro'], (0, 255, 0), 2)
                print(f"📍 Relativas -> X: {dx:.1f} Y: {dy:.1f} Z: {dz:.1f} | Dist: {distancia_3d:.1f}cm")
                
                cv2.putText(annotated_frame, f"Dist: {distancia_3d:.1f}cm", 
                            (c2['centro'][0], c2['centro'][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            last_prediction_time = current_time
        
        # Mostrar el frame procesado o el original si aún no hay proceso
        display_frame = annotated_frame if annotated_frame is not None else frame
        cv2.imshow('Robotica Vision - Cubos y Braccio', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()