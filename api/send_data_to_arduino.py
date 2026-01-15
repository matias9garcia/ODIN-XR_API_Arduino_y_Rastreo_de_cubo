import serial
import time
import requests
import json
import sys

API_URL = "http://127.0.0.1:5000/api/angulos_braccio"
PORT = "COM4"
BAUD = 9600

try:
    print(f"Conectando a Arduino en {PORT}...")
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    arduino.reset_input_buffer()
    arduino.reset_output_buffer()

    # Espera a que Arduino diga READY
    ready = arduino.readline().decode().strip()
    print("Arduino dijo:", ready)

except Exception as e:
    print("No se pudo abrir el puerto serial:", e)
    sys.exit(1)

print("Enviando datos a Arduino...\n")

while True:
    try:
        r = requests.get(API_URL)
        data = r.json()

        mensaje = f"{data['m1']},{data['m2']},{data['m3']},{data['m4']},{data['m5']},{data['m6']}\n"

        arduino.write(mensaje.encode())
        arduino.flush()

        print("Enviado ->", mensaje.strip())

        respuesta = arduino.readline().decode().strip()
        if respuesta:
            print("Respuesta ->", respuesta)

        time.sleep(0.5)

    except Exception as err:
        print("Error interno:", err)
        time.sleep(1)
