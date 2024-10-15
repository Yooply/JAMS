import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class DataType(Enum):
    TIME = 0,
    SPEED = 1,
    CPA = 3

# CO's standing orders
class COSO:
    def __init__(self, request_type, value):
        self.request_type = request_type
        self.value = value

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

def distance(p1, p2):
    x0,y0 = p1
    x1,y1 = p2
    return np.sqrt((x0-x1)**2+(y0-y1)**2)
    


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

    def distance(self, other):
        return distance(vec2coord(self.range, self.bearing), vec2coord(other.range, other.bearing))

    def to_coords(self):
        return vec2coord(self.range, self.bearing)

class ManeuveringBoard:
    def __init__(self, s_speed, s_course, scale=1) -> None:
        self.fig, self.ax = plt.subplots()
        self.scale = scale
        self.fixes = {}
        self.speed = s_speed
        self.course = s_course
        self.sx, self.sy = vec2coord(self.scale_speed(s_speed), s_course)
        self.draw_board()
        self.draw_dot(*vec2coord(self.scale_speed(s_speed),s_course), f"Speed Dot: {s_speed:.2f} kts, {s_course:.2f}ºT")
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
        self.ax.annotate(label, (x,y), fontweight="bold")

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
        self.speed_line_intercept = b_speed
        self.relative_dx = dX
        self.relative_dy = dY
        self.last_fix = f1

        x_course, y_course = self.get_circle_intercept(slope, b_course, 10000*self.scale, dX, dY, x0)
        x_speed_int, y_speed_int = self.get_circle_intercept(slope, b_speed, 10000*self.scale, dX, dY, x_speed)
        x_int, y_int = self.get_circle_intercept(slope, 0, 10000*self.scale, dX, dY)
        self.draw_vector_from(*coord2vec(x_course,y_course), x1,y1, color="c")
        self.draw_vector_from(*coord2vec(x_speed_int,y_speed_int), x_speed,y_speed, color="c")
        _, drm = coord2vec(x_int, y_int)
        self.draw_dot(x_int, y_int, f"DRM: {drm:.2f}ºT")
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
        dT = abs(f1.time-f2.time)
        # 3 minute rule
        if dT == 3:
            return dist/100
        else:
            return calculate_speed(dT, dist)

    def get_true_speed_course(self, vessel_id):
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()
        srm = self.get_srm(vessel_id)
        srm_dist = self.scale_speed(srm)
        x_speed, y_speed = vec2coord(self.scale_speed(self.speed), self.course)
        
        x = srm_dist*(1/(np.sqrt(1+self.relative_couse_slope**2)))
        y = srm_dist*(self.relative_couse_slope/(np.sqrt(1+self.relative_couse_slope**2)))
        if self.relative_dx < 0:
            x *= -1
            y *= -1
        x += x_speed
        y += y_speed
        mag,tbrg = coord2vec(x,y)
        mag = (mag / 1000*self.scale) * 5
        self.draw_dot(x,y, f"True Speed/Course: {mag:.2f} kts, {tbrg:.2f}ºT")
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
        self.draw_dot(x_int,y_int, f"CPA: {cpa_rng:.2f} yds, {cpa_brg:.2f}ºT")
        x0,y0 = vec2coord(f1.range, f1.bearing)
        dist = np.sqrt((x0-x_int)**2+(y0-y_int)**2)
        srm = self.get_srm(vessel_id)
        ttcpa = calculate_time(srm, dist)
        return cpa_rng, cpa_brg, ttcpa

    def solve_cpa(self, fix1, fix2):
        self.plot_fix(fix1, "contact")
        self.plot_fix(fix2, "contact")
        drm = self.draw_relative_lines("contact")
        srm = self.get_srm("contact")
        true_speed, true_course = self.get_true_speed_course("contact")
        cpa_rng, cpa_brng, t_to_cpa = self.get_time_brg_rng_cpa("contact")
        print("------ Solving Closest Point of Approach (CPA) -------")
        print("DRM:               ", drm)
        print("SRM:               ", srm)
        print("True Speed:        ", true_speed)
        print("True Course:       ", true_course)
        print("CPA Range:         ", cpa_rng)
        print("CPA Bearing:       ", cpa_brng)
        print("Time to CPA (RAW): ", t_to_cpa)
        print("Time to CPA (ADJ): ", t_to_cpa+fix2.time)

    def new_fix_from_time(self, time, vessel_id):
        if vessel_id not in self.fixes or len(self.fixes[vessel_id]) < 2:
            raise IndexError()
        srm = self.get_srm(vessel_id)
        dist = calculate_dist(time,srm)
        last_fix = self.fixes[vessel_id][-1]
        
        x0,y0 = vec2coord(last_fix.range, last_fix.bearing)

        x = dist*(1/(np.sqrt(1+self.relative_couse_slope**2))) + x0
        y = dist*(self.relative_couse_slope/(np.sqrt(1+self.relative_couse_slope**2))) + y0
        rng, brng = coord2vec(x,y)
        new_fix = Fix(brng,rng, last_fix.time+time)
        self.plot_fix(new_fix, vessel_id)

    def find_tangent_circle_points(self, radius, x, y):
        a = (x**2+y**2)
        b = -1*(2*radius**2*x)
        c = -1*(radius**2*y**2) + radius**4
        x0 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        x1 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
        y0 = (radius**2-x*x0)/y
        y1 = (radius**2-x*x1)/y
        return [(x0,y0), (x1,y1)]

    def solve_avoidance(self, radius, time_since_last_fix, vessel_id, use_closer_point=False):
        self.new_fix_from_time(time_since_last_fix, vessel_id)
        ring = plt.Circle((0,0), radius, color="r", fill=False)
        self.coso_radius = radius
        self.ax.add_patch(ring)
        last_fix = self.fixes[vessel_id][-1]
        x,y=vec2coord(last_fix.range, last_fix.bearing); 
        points = self.find_tangent_circle_points(radius, x,y)
        # print(points)
        speed_dot = (self.sx, self.sy) 
        dist1 = distance(points[0], speed_dot)
        dist2 = distance(points[1], speed_dot)
        if use_closer_point:
            if (dist1 > dist2):
                self.ax.plot([x,points[1][0]], [y, points[1][1]])
            else:
                self.ax.plot([x,points[0][0]], [y, points[0][1]])
        else:
            if (dist1 > dist2):
                self.ax.plot([x,points[0][0]], [y, points[0][1]])
            else:
                self.ax.plot([x,points[1][0]], [y, points[1][1]])




    
    def solve_stationing_speed(self, speed):
        ring = self.scale_speed(speed)
        x = (ring-self.speed_line_intercept)/self.relative_couse_slope
        _, tbrng = coord2vec(x,ring)
        x0,y0 = vec2coord(self.scale_speed(self.speed), self.course)
        srm = (np.sqrt((ring-y0)**2+(x-x0)**2) / 1000)*5
        self.draw_dot(x,ring, f"True Speed/Course: {tbrng:.2f}ºT @ {speed:.2f} kts")
        return tbrng, srm
    
    def solve_stationing_time(self, srm):
        x_speed,y_speed = self.sx, self.sy
        srm_dist = self.scale_speed(srm)
        
        x = srm_dist*(1/(np.sqrt(1+self.relative_couse_slope**2)))
        y = srm_dist*(self.relative_couse_slope/(np.sqrt(1+self.relative_couse_slope**2)))
        if self.relative_dx < 0:
            x *= -1
            y *= -1
        x += x_speed
        y += y_speed
        mag,tbrg = coord2vec(x,y)
        mag = (mag / 1000*self.scale) * 5
        self.draw_dot(x,y, f"True Speed/Course: {mag:.2f} kts, {tbrg:.2f}ºT")
        return mag,tbrg



    def solve_stationing(self, starting_point: Fix, ending_point: Fix, orders: COSO):
        self.plot_fix(starting_point, "self")
        self.plot_fix(ending_point, "self")
        self.draw_relative_lines("self")
        srm = 0.0
        tbrng = 0.0
        tspeed = 0.0
        time_to_station = 0.0
        dist = starting_point.distance(ending_point)
        match orders.request_type:
            case DataType.SPEED:
                tspeed = orders.value
                tbrng, srm = self.solve_stationing_speed(tspeed)
                time_to_station = calculate_time(srm, dist)
            case DataType.TIME:
                time_to_station = orders.value
                srm = calculate_speed(time_to_station, dist)
                tspeed, tbrng = self.solve_stationing_time(srm)
            case _:
                raise ValueError("CO's standing orders are not valid for this problem type")
        print("----------------- Solving Stationing -----------------")
        print("SRM:                   ", srm)
        print("True Speed:            ", tspeed)
        print("True Course:           ", tbrng)
        print("Time to station (RAW): ", time_to_station)
        print("Time to station (ADJ): ", time_to_station+starting_point.time)
        




a = ManeuveringBoard(10, 90)

fix1 = Fix(60, 10000, 1100)
fix2 = Fix(58, 8000, 1103)

a.solve_cpa(fix1,fix2)
a.solve_avoidance(3500, 3,"contact")

"""
b = ManeuveringBoard(25,20)
start = Fix(140, 7000, 1100)
end = Fix(0,0,-1)
orders = COSO(DataType.TIME, 24.4)
b.solve_stationing(start, end, orders)
"""

plt.show()
