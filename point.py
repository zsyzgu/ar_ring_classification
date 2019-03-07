class Point:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(a, b):
        return Point(a.x + b.x, a.y + b.y, a.z + b.z)
    
    def __sub__(a, b):
        return Point(a.x - b.x, a.y - b.y, a.z - b.z)
    
    def __mul__(a, k):
        return Point(a.x * k, a.y * k, a.z * k)
    
    def __div__(a, k):
        return Point(a.x / k, a.y / k, a.z / k)
    
    def __str__(a):
        return '(' + str(a.x) + ', ' + str(a. y) + ', ' + str(a.z) + ')'

    def unit(a):
        return a / a.module()
    
    def dot(a, b):
        return a.x * b.x + a.y * b.y + a.z * b.z
    
    def mul(a, b):
        return Point(a.y * b.z - b.y * a.z, b.x * a.z - a.x * b.z, a.x * b.y - b.x * a.y)

    def dist2(a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2

    def dist(a, b):
        return a.dist2(b) ** 0.5

    def module2(a):
        return a.x ** 2 + a.y ** 2 + a.z ** 2
    
    def module(a):
        return a.module2() ** 0.5
