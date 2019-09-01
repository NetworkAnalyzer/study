# -*- coding: UTF-8 -*-

class Object:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c_x = 0
        self.c_y = 0
        self.compactness = self.compactness()
        self.hwr = self.hwr()
        self.image = ""

    def compactness(self):
        area = self.h * self.w
        prim = 2 * self.h + 2 * self.w

        return float(area) / prim**2

    def hwr(self):
        return float(self.h) / self.w

    def velocity(self):
        return

if __name__ == "__main__":
    object = Object(23, 25, 34, 43)
    print(object.compactness)
    print(object.hwr)
