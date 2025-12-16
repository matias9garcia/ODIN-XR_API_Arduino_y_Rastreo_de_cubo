import cv2
import time
from ultralytics import YOLO

def seleccionar_camara():
    """
    Busca y selecciona la primera cámara USB disponible.
    """
    print("🔍 Buscando cámaras disponibles...")
    camaras_disponibles = []
    
    # Rango de 0 a 4 para buscar índices de cámara (puedes ampliarlo si tienes más cámaras)
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"✅ Cámara detectada en índice {i}")
                camaras_disponibles.append(i)
            cap.release()
            
    if not camaras_disponibles:
        print("⚠️ No se detectó ninguna cámara. Verifica la conexión USB o los drivers.")
        return -1 
    
    camara_seleccionada = min(camaras_disponibles)
    print(f"🎥 Usando cámara en índice: {camara_seleccionada}")
    return camara_seleccionada

def main():
    """
    Carga el modelo YOLO y realiza la detección a una frecuencia limitada (cada 0.5 segundos).
    """
    try:
        model = YOLO("my_model.pt")
        print("✅ Modelo YOLO 'my_model.pt' cargado exitosamente.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo 'my_model.pt': {e}")
        print("Asegúrate de que el archivo del modelo esté en la ruta correcta.")
        return

    camera_index = seleccionar_camara()
    if camera_index == -1:
        print("Saliendo del programa porque no se pudo encontrar una cámara.")
        return

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir la cámara en el índice {camera_index}.")
        print("Asegúrate de que la cámara no esté siendo utilizada por otra aplicación.")
        return

    # --- Configuración de Temporización ---
    PREDICTION_INTERVAL = 0.2  # Intervalo de tiempo en segundos (0.5s)
    last_prediction_time = time.time()  # Inicializa el tiempo de la última predicción
    # -------------------------------------

    # Variable para almacenar el frame ANOTADO de la última predicción
    annotated_frame = None 

    print("\n--- Detección en tiempo real iniciada (Predicción limitada a 0.5s) ---")
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: No se pudo leer el frame de la cámara. Saliendo...")
            break

        current_time = time.time()
        
        # --- Lógica de Predicción con Temporizador ---
        if (current_time - last_prediction_time) >= PREDICTION_INTERVAL:
            
            # 1. Realiza la inferencia SÓLO si han pasado 0.5 segundos
            results = model(frame, stream=True)
            
            # 2. Obtiene el frame anotado y lo guarda
            for r in results:
                annotated_frame = r.plot()
                break # Solo necesitamos un frame anotado
            
            # 3. Actualiza el tiempo de la última predicción
            last_prediction_time = current_time
        
        # --- Mostrar el Frame ---
        # Si ya se ha realizado una predicción, muestra el último frame anotado.
        # Si no, muestra el frame crudo para mantener el video fluido mientras se espera.
        if annotated_frame is not None:
            cv2.imshow('YOLOv8 Live Detection (0.5s Update)', annotated_frame)
        else:
            # Esto se ejecutará en la primera iteración antes de la primera predicción
             cv2.imshow('YOLOv8 Live Detection (0.5s Update)', frame)


        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la cámara y destruye todas las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Detección finalizada ---")

if __name__ == "__main__":
    main()