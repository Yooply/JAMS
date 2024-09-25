import matplotlib.pyplot as plt
import numpy as np

class ManeuveringBoard:
    def __init__(self, s_speed, s_course) -> None:
        self.fig, self.ax = plt.subplots()
        self.draw_board()
        self.draw_dot(*self.vec2coord(self.scale_speed(s_speed),s_course), "Speed Dot")
        self.ax.set(xlim=(-12, 12), xticks=np.arange(0,0),ylim=(-12,12), yticks=np.arange(0, 0))
    
    def scale_speed(self, speed):
        return speed / 5

    def vec2coord(self, magnitude, bearing) -> tuple[float, float]:
        bearing = (bearing + 90)
        b_rads = np.radians(bearing)
        x = magnitude * np.cos(b_rads) * -1; # Reflect about y-axis as we want clockwise angles
        y = magnitude * np.sin(b_rads);
        return x,y

    def draw_vector_from(self, magnitude, bearing, x0=0, y0=0, label="", linestyle="-", color="g"):
        x,y = self.vec2coord(magnitude, bearing)
        self.ax.plot([x0, x],[y0,y],color=color, linestyle=linestyle, label=label)
    
    def draw_dot(self, x, y, label):
        self.ax.plot(x,y, "ro",label=label)

    def draw_board(self):
        circles = []
        for i in range(1,11):
            ring = plt.Circle((0,0), i, color="g", fill=False)
            self.ax.add_patch(ring)
            circles.append(ring)
        for i in range(0,360,10):
            x,y = self.vec2coord(1, i)
            self.draw_vector_from(10, i, x, y, linestyle="--")
            self.ax.text(x*11,y*11, "0/360" if i == 0 else str(i), horizontalalignment="center")
        return circles




ManeuveringBoard(15, 0)
plt.show()
