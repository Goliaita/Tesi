


class Reparametrize:

    def __init__(self):
        super()

        
    fx0 = 519.299
    fy0 = 511.099
    cx0 = 323.50
    cy0 = 238.00
    s0  = 8000


    fx1 = 637.735
    fy1 = 637.735
    cx1 = 645.414
    cy1 = 349.205
    s1  = 1000

    def compute_dimension(self, x, y, z, id=0):
        if id == 0:
            fx = self.fx0
            fy = self.fy0
            cx = self.cx0
            cy = self.cy0
            s  = self.s0
        else:
            fx = self.fx1
            fy = self.fy1
            cx = self.cx1
            cy = self.cy1
            s  = self.s1

        Z = z / s
        X = (x - cx) * Z / fx 
        Y = (y - cy) * Z / fy

        return X, Y, Z
