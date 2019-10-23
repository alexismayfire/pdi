import cv2

cap = cv2.VideoCapture(0)

# Confere se a webcam pode ser acessada
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Input", frame)

    c = cv2.waitKey(1)
    # Valor ASCII do Esc
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
