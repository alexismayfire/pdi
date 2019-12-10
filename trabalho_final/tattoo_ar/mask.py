class Line:
    def __init__(self, xy_start, xy_end):
        self.start = xy_start
        self.end = xy_end

        # Para garantir que x1/y1 será sempre a menor coordenada
        # E x2/y2 sempre será a maior
        if xy_end[0] > xy_start[0]:
            self.x1 = xy_start[0]
            self.x2 = xy_end[0]
        else:
            self.x1 = xy_end[0]
            self.x2 = xy_start[0]

        if xy_end[1] > xy_start[1]:
            self.y1 = xy_start[1]
            self.y2 = xy_end[1]
        else:
            self.y1 = xy_end[1]
            self.y2 = xy_start[1]

        self.x = (self.x1 + self.x2) // 2
        self.y = (self.y1 + self.y2) // 2

        self.x_deviation = abs(self.x1 - self.x2)
        self.y_deviation = abs(self.y1 - self.y2)

    def __str__(self):
        return f'[({self.x1}, {self.y1}), ({self.x2}, {self.y2})]'

    def horizontal(self, max_deviation) -> bool:
        return max_deviation > self.y_deviation and max_deviation < self.x_deviation

    def vertical(self, max_deviation) -> bool:
        return max_deviation > self.x_deviation and max_deviation < self.y_deviation


class Shape:
    # Line só por typehint
    left: Line = None
    right: Line = None
    bottom: Line = None

    @classmethod
    def detected(cls):
        if cls.left and cls.right and cls.bottom:
            return True

        return False

    @classmethod
    def size(cls):
        width = abs(cls.right.x - cls.left.x)

        if cls.right.y1 > cls.left.y1:
            height = abs(cls.right.y1 - cls.bottom.y2)
        else:
            height = abs(cls.left.y1 - cls.bottom.y2)
        
        return (width, height)

    @classmethod
    def y_start(cls):
        return (cls.left.y1 + cls.right.y1) // 2

    @classmethod
    def y_end(cls):
        vertical_y = (cls.left.y2 + cls.right.y2) // 2
        if vertical_y > cls.bottom.y:
            return vertical_y
        else:
            return cls.bottom.y


def get_mask_coordinates(frame):
    """
    Gera as coordenadas para desenhar a máscara na tela

    """

    height, width, _ = frame.shape
    
    x = int(width * 0.35)
    x = int(x + (x * 0.25))

    y = int(height * 0.4)

    # Offset entre a linha de baixo com as linhas laterais
    offset = 20

    col_left = x
    col_right = width - x
    row_top = y
    row_bottom = height - y

    left = Line(
        (col_left, row_top),
        (col_left, row_bottom - offset)
    )
    right = Line(
        (col_right, row_top),
        (col_right, row_bottom - offset)
    )

    bottom = Line(
        (col_left, row_bottom),
        (col_right, row_bottom)
    )

    return (left, right, bottom)


class Coordinates:
    class __Coordinates:
        def __init__(self, frame):
            self.frame = frame
        
        def __str__(self):
            return repr(self)

    instance = None

    def __init__(self, frame):
        if not Coordinates.instance:
            Coordinates.instance = Coordinates.__Coordinates(arg)
        else:
            Coordinates.instance.frame = frame
        
    def __getattribute__(self, name):
        return getattr(self.instance, name)