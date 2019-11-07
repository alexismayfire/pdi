import cv2


class Mask:
    def __init__(self, frame):
        self.frame = frame
        height, width, channels = frame.shape
        x = width // 3  # 33% da largura
        self.x = int(x + (x // 5))
        self.y = height // 3  # 33% da altura

    def square():
        # Linha da esquerda
        cv2.line(self.frame, (x, y), (x, height - y), (255, 0, 0), 3)
        # Linha da direita
        cv2.line(self.frame, (width - x, y), (width - x, height - y), (255, 0, 0), 3)
        # Linha inferior
        cv2.line(
            self.frame,
            (x + 10, height - y),
            (width - x - 10, height - y),
            (255, 0, 0),
            3,
        )

        return self.frame

    def x():
        # Linha da esquerda
        cv2.line(self.frame, (x, y), (width - x, height - y), (255, 0, 0), 3)
        # Linha da direita
        cv2.line(self.frame, (width - x, y), (x, height - y), (255, 0, 0), 3)

        return self.frame
