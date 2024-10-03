import matplotlib.pyplot as plt
import numpy as np

def vec2coord(magnitude, bearing, x0=0, y0=0) -> tuple[float, float]:
    bearing = (bearing + 90)
    b_rads = np.radians(bearing)
    x = (magnitude * np.cos(b_rads) * -1) + x0; # Reflect about y-axis as we want clockwise angles
    y = (magnitude * np.sin(b_rads)) + y0;
    return x,y
    
def coord2vec(x, y) -> tuple[float, float]:
    magnitude = np.sqrt(x**2 + y**2)
    # TODO: Fix issue where tan(theta) -> +/- infinity
    bearing = (np.degrees(2*np.arctan(y/((x*-1)+magnitude)))-90)
    if (bearing < 0):
        bearing += 360
    return magnitude, bearing

# 1 knot = 1 nautical mile / hr
# 1 nautical mile = 2025.372 yards

def calculate_dist(time, speed):
    return speed*(time/60)*2025.372

def calculate_time(speed, distance):
    return (distance/2025.372)*60/speed

def calculate_speed(time, distance):
    return distance/2025.372*60/time

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
        self.speed = s_speed
        self.course = s_course
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

    def draw_relative_lines(self, vessel_id) -> float:
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()
        
        f1 = self.fixes[vessel_id][-1]
        f2 = self.fixes[vessel_id][-2]

        x0,y0 = vec2coord(f1.range, f1.bearing)
        x1,y1 = vec2coord(f2.range, f2.bearing)

        slope = (y0-y1) / (x0-x1)
        b_course = y0 - slope*x0
        dX = x0-x1
        dY = y0-y1
        x_speed, y_speed = vec2coord(self.scale_speed(self.speed), self.course)
        b_speed = y_speed - slope*x_speed
        self.relative_couse_slope = slope
        self.relative_couse_intercept = b_course
        self.relative_dx = dX
        self.relative_dy = dY
        self.last_fix = f1

        x_course, y_course = self.get_circle_intercept(slope, b_course, 10000*self.scale, dX, dY, x0)
        x_speed_int, y_speed_int = self.get_circle_intercept(slope, b_speed, 10000*self.scale, dX, dY, x_speed)
        x_int, y_int = self.get_circle_intercept(slope, 0, 10000*self.scale, dX, dY)
        self.draw_dot(x_int, y_int, "DRM")
        self.draw_vector_from(*coord2vec(x_course,y_course), x1,y1, color="c")
        self.draw_vector_from(*coord2vec(x_speed_int,y_speed_int), x_speed,y_speed, color="c")
        _, drm = coord2vec(x_int, y_int)
        return drm
        

    def get_circle_intercept(self, slope, intercept, radius, dX, dY, lfx=0) -> tuple[float, float]:
        # Intersection of a centered circle and line
        # x^2 + y^2 = r^2
        # x^2 + (mx+b)^2=r^2
        # x^2 + (mx)^2 + 2(bmx) + b^2 - r^2 = 0
        # (m^2+1)x^2 + 2bmx + (b^2-r^2) = 0
        # y = m((-2bm +/- sqrt((2bm)^2-4(m^2+1)(b^2-r^2)))/(2(m^2+1)))+b
        try:
            x0 = (((-2*intercept*slope) + ((2*intercept*slope)**2-4*(slope**2+1)*(intercept**2-radius**2))**0.5)/(2*(slope**2+1)))
            x1 = (((-2*intercept*slope) - ((2*intercept*slope)**2-4*(slope**2+1)*(intercept**2-radius**2))**0.5)/(2*(slope**2+1)))
        except ZeroDivisionError:
            if (dY > 0):
                return lfx, radius
            elif (dY < 0):
                return lfx, radius*-1
            else:
                raise ValueError("Vessel is not moving relative to center")

        if (dX > 0):
            x = max(x0,x1)
        else:
            x = min(x0,x1)

        return x, slope*x+intercept

    def get_srm(self, vessel_id):
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()

        f1 = self.fixes[vessel_id][-1]
        f2 = self.fixes[vessel_id][-2]

        x0,y0 = vec2coord(f1.range, f1.bearing)
        x1,y1 = vec2coord(f2.range, f2.bearing)
        
        dist = ((x0-x1)**2+(y0-y1)**2)**0.5
        
        # 3 minute rule
        if abs(f1.time-f2.time) == 3:
            return dist/100

    def get_true_speed_course(self, vessel_id):
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()
        srm = self.get_srm(vessel_id)
        srm_dist = self.scale_speed(srm)
        x_speed, y_speed = vec2coord(self.scale_speed(self.speed), self.course)
        
        x = srm_dist*(1/(np.sqrt(1+self.relative_couse_slope**2))) + x_speed
        y = srm_dist*(self.relative_couse_slope/(np.sqrt(1+self.relative_couse_slope**2))) + y_speed
        mag,tbrg = coord2vec(x,y)
        mag = (mag / 1000*self.scale) * 5
        self.draw_dot(x,y, "True Speed")
        return mag,tbrg

    def get_time_brg_rng_cpa(self, vessel_id):
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()
        
        perpendicular_slope = -1*(self.relative_couse_slope**-1)
        f1 = self.fixes[vessel_id][-1]
        # m1x+b1 = m2x+b2
        # (m1-m2)x + (b1-b2) = 0
        # x(m1-m2) = b2-b1
        # x=(b2-b1)/(m1-m2)
        x_int = self.relative_couse_intercept/(perpendicular_slope-self.relative_couse_slope)
        y_int = perpendicular_slope*x_int
        cpa_rng,cpa_brg = coord2vec(x_int,y_int)
        self.draw_dot(x_int,y_int, "CPA")
        x0,y0 = vec2coord(f1.range, f1.bearing)
        dist = np.sqrt((x0-x_int)**2+(y0-y_int)**2)
        srm = self.get_srm(vessel_id)
        ttcpa = calculate_time(srm, dist)
        return cpa_rng, cpa_brg, ttcpa


        

a = ManeuveringBoard(11, 40)
        

fix1 = Fix(321, 6500, 0)
fix2 = Fix(330, 5000, 3)

a.plot_fix(fix1, "a")
a.plot_fix(fix2, "a")
drm = a.draw_relative_lines("a")
#a.draw_vector_from(10000, drm ,color="b")
print("DRM: ", drm)
srm = a.get_srm("a")
print("SRM: ", srm)
true_speed, true_course = a.get_true_speed_course("a")
print("True Speed", true_speed)
print("True Course", true_course)

cpa_range, cpa_bearing, time_to_cpa = a.get_time_brg_rng_cpa("a")

print("CPA Range", cpa_range)
print("CPA Bearing", cpa_bearing)
print("Time To CPA", time_to_cpa)



# a.draw_vector_from(5000, 90, -5000, 5000, color="r")
plt.show()
