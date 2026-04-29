import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen
img = cv2.imread("imagen.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Crear detector de QR
qr_detector = cv2.QRCodeDetector()

# Detectar varios QR
retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(img)

# Copia para dibujar resultados
img_result = img_rgb.copy()

if retval:
    print(f"QR detectados: {len(points)}")

    for i, qr_points in enumerate(points):
        qr_points = qr_points.astype(int)

        # Dibujar contorno del QR
        cv2.polylines(
            img_result,
            [qr_points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3
        )

        # Centro del QR
        cx = int(np.mean(qr_points[:, 0]))
        cy = int(np.mean(qr_points[:, 1]))

        cv2.circle(img_result, (cx, cy), 6, (0, 255, 0), -1)

        print(f"\nQR {i+1}")
        print("Contenido:", decoded_info[i])
        print("Esquinas:")
        print(qr_points)
        print("Centro:", (cx, cy))

else:
    print("No se han detectado QR.")

plt.figure(figsize=(10, 10))
plt.imshow(img_result)
plt.axis("off")
plt.show()