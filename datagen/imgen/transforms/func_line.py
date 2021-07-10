import math
import random
import numpy as np
import cv2 as cv

__all__ = ['draw_random_lines',]

# Rotation matrix function
def rotate_matrix(x, y, angle, x_shift=0, y_shift=0):
    x, y = x - x_shift, y - y_shift
    angle = math.radians(angle)
    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift
    xr,yr = int(xr), int(yr)
    return xr, yr


def split_indices(num, factor):
    k = math.floor(num * factor)
    indices = list(range(num))
    if k>0:
        val_indices = random.sample(indices, k=k)
        trn_indices = list(set(indices) - set(val_indices))
    else:
        val_indices = None
        trn_indices = indices
    return trn_indices, val_indices


def random_point(xmin=0, xmax=640, ymin=0, ymax=480):
    x, y = random.randint(xmin,xmax), random.randint(ymin,ymax)
    pt = (x, y)
    return pt

def random_multi_points(num, angle=30, size=10, xmin=0, xmax=640, ymin=0, ymax=480):
#     points = [random_point(ymax=h, xmax=w) for i in range(num)] 
    if num==2:
        pts = random_point(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        xs,ys = pts
        xm, ym = xs + size, ys + size
        
        if random.randint(0,1):
            angle = random.randint(0, angle) * random.choice([-1,1]) 
            xm,ym = rotate_matrix(xm,ym, angle=angle)
        ptm = xm, ym
        points = []
        points.append(pts)
        points.append(ptm)
    elif num>2:
        pts = random_point(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        xs, ys = pts
        
        xm, ym = xs + size, ys + size
        if random.randint(0,1):
            angle = random.randint(0,angle) * random.choice([-1,1]) 
            xm,ym = rotate_matrix(xm,ym, angle=angle)
        ptm = xm, ym

        points = []
        points.append(pts)
        nxs, nxm, nys, nym = min(xs, xm), max(xs,xm), min(ys, ym), max(ys, ym)  
        for i in range(num-2):
            x,y = ptb = random_point(xmin=nxs, xmax=nxm, ymin=nys, ymax=nym)
            angle = random.randint(0,10)
            x,y = rotate_matrix(x,y, angle=angle)
            ptb = x,y
            points.append(ptb)
        points.append(ptm)
    else:
        raise Exception("num cannot less than 2")
        
    return points

def random_multi_std_line(num, size=10, angle=30, min_point=2, max_point=5, h=480, w=640):
    num_point = random.randint(min_point, max_point)
    lines = [random_multi_points(num_point, size=size, angle=angle, ymax=h, xmax=w) for i in range(num)]
    lines = sorted(lines)
    return lines


def random_multi_line(num, size=10, angle=13, curve_factor=0.2, min_point=2, max_point=3):
    points = random_multi_std_line(num, size=size, angle=angle, 
                                   min_point=min_point, 
                                   max_point=max_point)
    
    str_indices, crv_indices = split_indices(num, factor=curve_factor)

    if crv_indices:
        for idx in range(len(points)):
            if idx in crv_indices:
                points[idx] = convert2curve(points[idx], num=30).tolist()
    return points


def convert2curve(point, start=50, stop=100, num=50):
    point = np.array(point)
    x = point.reshape(-1,2)[:,0]
    y = point.reshape(-1,2)[:,1]
    start = np.min([x.min(),y.min()])
    stop = np.max([x.max(),y.max()])
    # Initilaize y axis
    lspace = np.linspace(start=start, stop=stop, num=num)
    
    #calculate the coefficients.    
    z = np.polyfit(x, y, 2)
    #calculate x axis
    line_fitx = (z[0]*lspace**2) + (z[1]*lspace) + z[2]
    verts = np.array(list(zip(lspace.astype(int), line_fitx.astype(int))))
    
    return verts

def line_random_color(index=None):
    colors = [(0,0,0),(0,0,255),(255,0,0),(0,255,0)]
    choice = random.choices(population=colors,weights=[0.8, 0.3, 0.15, 0.05], k=1)
    choiced = choice[0]
    if index:
        choiced = colors[index]
    return choiced

def draw_lines(image, points, color=(0,0,0), thickness=1, line_type=cv.LINE_AA, random_color=False, color_mode='single'):
    img = image.copy()
    if random_color==True and color_mode=='single':
        color = line_random_color()
    for line in points:
        pts = np.array(line, dtype=np.int32)
        if random_color==True and color_mode=='distributed':
            color = line_random_color()
        cv.polylines(img, [pts], False, color=color, thickness=thickness, lineType=line_type)
    return img

def draw_random_lines(image, num_line, 
                      line_size=50, line_thickness=1,
                      angle=13, curve_factor=0.2, 
                      min_point=2, max_point=3,
                      randomize_line_color=False, line_color_mode='single'):
    
    points = random_multi_line(num_line, size=line_size, 
                               angle=angle, curve_factor=curve_factor,
                               min_point=min_point, max_point=max_point)
    # points = np.array(points)
    
    image_lines = draw_lines(image, points, 
                             thickness=line_thickness,
                             random_color=randomize_line_color, 
                             color_mode=line_color_mode)
    return image_lines