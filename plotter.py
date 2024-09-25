import matplotlib.pyplot as plt
import numpy as np

def vec2coord(magnitude, bearing) -> tuple[float, float]:
    bearing = (bearing + 90)
    b_rads = np.radians(bearing)
    x = magnitude * np.cos(b_rads) * -1; # Reflect about y-axis as we want clockwise angles
    y = magnitude * np.sin(b_rads);
    return x,y
    
def coord2vec(x, y) -> tuple[float, float]:
    magnitude = np.sqrt(x**2 + y**2)
    bearing = (np.degrees(2*np.arctan(y/((x*-1)+magnitude)))-90)
    return magnitude, bearing


class Fix:
    def __init__(self, brng, rng, time):
        self.bearing = brng
        self.range = rng
        self.time = time

    def plot(self, ax):
        ax.plot(*vec2coord(self.range, self.bearing), color="r", label=str(self.time))

class ManeuveringBoard:
    def __init__(self, s_speed, s_course, scale=1) -> None:
        self.fig, self.ax = plt.subplots()
        self.scale = scale
        self.fixes = {}
        self.draw_board()
        self.draw_dot(*vec2coord(self.scale_speed(s_speed),s_course), "Speed Dot")
        self.ax.set(xlim=(-12000, 12000), xticks=np.arange(0,0),ylim=(-12000,12000), yticks=np.arange(0, 0))
    
    def scale_speed(self, speed):
        return (speed / 5)*1000;
    
    def scale_distance(self, distance):
        return distance / (self.scale)

    def draw_vector_from(self, magnitude, bearing, x0=0, y0=0, label="", linestyle="-", color="g"):
        x,y = vec2coord(magnitude, bearing)
        self.ax.plot([x0, x],[y0,y],color=color, linestyle=linestyle, label=label)
    
    def draw_dot(self, x, y, label):
        self.ax.plot(x,y, "ro",label=label)

    def draw_board(self):
        circles = []
        for i in range(1,11):
            ring = plt.Circle((0,0), i*1000, color="g", fill=False)
            self.ax.add_patch(ring)
            circles.append(ring)
        for i in range(0,360,10):
            x,y = vec2coord(1000, i)
            self.draw_vector_from(10000, i, x, y, linestyle="--")
            self.ax.text(x*11,y*11, "0/360" if i == 0 else str(i), horizontalalignment="center")
        return circles

    def plot_fix(self, fix, vessel_id):
        if vessel_id not in self.fixes:
            self.fixes[vessel_id] = []
        self.fixes[vessel_id].append(fix)
        self.draw_dot(*vec2coord(self.scale_distance(fix.range), fix.bearing), label=str(fix.time))

    def draw_relative_speed_line(self, vessel_id):
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()
        
        f1 = self.fixes[vessel_id][-1]
        f2 = self.fixes[vessel_id][-2]

        x0,y0 = vec2coord(f1.range, f1.bearing)
        x1,y1 = vec2coord(f2.range, f2.bearing)

        

a = ManeuveringBoard(15, 0)

fix1 = Fix(220, 9000, 1000)
fix2 = Fix(222, 8100, 1003)

a.plot_fix(fix1, "a")
a.plot_fix(fix2, "a")

plt.show()
