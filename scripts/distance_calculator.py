import cv2
import time
import math
from ultralytics import YOLO

# --- Configuración de Parámetros ---
PXM_RATIO = 0.1  # Factor de conversión: cuántos cm es 1 píxel (ajustar según tu setup)
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
    annotated_frame = None 

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        
        if (current_time - last_prediction_time) >= 0.2:
            results = model(frame)
            cubos_data = [] # Guardaremos (centro_x, centro_y, area)
            annotated_frame = frame.copy()

            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if model.names[class_id] == "cubo_morado":
                        b = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = b
                        
                        # Calcular centro y área (para profundidad Z)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        area = (x2 - x1) * (y2 - y1)
                        
                        cubos_data.append({'centro': (cx, cy), 'area': area, 'box': (int(x1), int(y1), int(x2), int(y2))})
                        
                        # Dibujo básico
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)

            # --- Cálculo de Coordenadas Relativas ---
            if len(cubos_data) >= 2:
                # Cubo 1 (Referencia/Origen) y Cubo 2 (Objetivo)
                c1 = cubos_data[0]
                c2 = cubos_data[1]
                
                # 1. Delta X y Delta Y en cm
                dx = (c2['centro'][0] - c1['centro'][0]) * PXM_RATIO
                dy = (c1['centro'][1] - c2['centro'][1]) * PXM_RATIO # Y invertida en imagen
                
                # 2. Delta Z (Estimación por diferencia de áreas)
                # Si Area2 < Area1, el cubo 2 está más lejos (Z positivo)
                # Usamos raíz cuadrada porque el área crece al cuadrado de la distancia
                relacion_z = math.sqrt(c1['area'] / c2['area'])
                dz = (relacion_z - 1.0) * 10 # Multiplicador arbitrario para escala en cm
                
                # 3. Distancia Euclídea 3D
                distancia_3d = math.sqrt(dx**2 + dy**2 + dz**2)

                # Dibujar línea y datos
                cv2.line(annotated_frame, c1['centro'], c2['centro'], (0, 255, 0), 2)
                
                print(f"📍 Relativas al Cubo 1 -> X: {dx:.2f}cm, Y: {dy:.2f}cm, Z: {dz:.2f}cm | Total: {distancia_3d:.2f}cm")
                
                cv2.putText(annotated_frame, f"D3D: {distancia_3d:.1f}cm", c2['centro'],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            last_prediction_time = current_time
        
        if annotated_frame is not None:
            cv2.imshow('Cálculo X Y Z Relativo', annotated_frame)
        else:
            cv2.imshow('Cálculo X Y Z Relativo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()