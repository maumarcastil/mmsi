import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.ticker import AutoMinorLocator


def read_and_clean_csv(path):
    data = pd.read_csv(path)
    notnull = pd.notnull(data["latitude"])
    data = data[notnull]
    
    ### obtenemos todos los barcos ###
    #info = data
    
    ### obtenemos uno o mas barcos especificos ###
    #info = data[(data.mmsi == 0)]
    info = data[(data.mmsi == 0) | (data.mmsi == 5853) | (data.mmsi == 258511000)]
    return info


### Creando los puntos en grafica ###
def create_polygon():
    coords = [(18.00548,-88.29161),(19.17153,-87.92393), (20.07163,-87.69908),
           (20.99716,-86.96372), (21.17102,-86.89854), (21.28327,-86.92124),
           (22.59117,-83.65847), (22.75036,-81.61381), (22.15346,-80.96375),
           (21.81945,-79.78635), (21.55837,-78.64004), (21.09196,-78.37263),
           (20.77605,-77.82145), (20.45948,-76.87476), (19.90474,-77.5899),
           (20.12257,-74.43184), (19.67372,-72.89676), (18.82608,-72.26106),
           (18.42148,-72.53647), (18.5264,-73.61899), (18.60339,-73.95005),
           (18.71472,-74.11009), (18.58823,-74.29997), (18.45643,-74.369),
           (18.34585,-74.27159), (18.2869,-74.00828), (18.07996,-73.79973),
           (18.34177,-73.58009), (18.26921,-73.21759), (18.17552,-72.75619),
           (18.27999,-72.13547), (18.21222,-71.82511), (18.00602,-71.60402),
           (17.69362,-71.34515), (18.17509,-70.99599), (18.49333,-70.55714),
           (18.27186,-70.40363), (18.54163,-69.46995), (18.32734,-68.57463),
           (18.34006,-67.0668), (18.1476,-67.05376), (18.0124,-67.05171),
           (18.01559,-66.08286), (18.04512,-65.78572), (18.26253,-65.56548),
           (18.43278,-64.86734), (18.40321,-63.5295), (18.08582,-62.921),
           (17.66539,-61.63897), (16.1507,-61.21768), (15.80188,-61.22222),
           (15.45246,-61.31465), (14.66712,-61.07749), (13.91109,-60.84905), 
           (13.1687,-59.58127), (11.2817,-60.39757), (10.82477,-61.02545), 
           (10.80352,-61.28404), (10.74703,-61.45179), (10.60271,-61.4203), 
           (10.47453,-61.44375), (10.29369,-61.41375), (10.23567,-61.48938), 
           (10.19115,-61.65566), (10.13041,-61.80133), (10.10003,-61.8467), 
           (10.12269,-61.71008), (10.1124,-61.05675), (9.837,-61.48008), 
           (9.90131,-61.59824), (9.81439,-61.54745), (9.80612,-61.67861),
           (9.7816,-61.80154), (9.96731,-62.04601), (10.01686,-62.17374), 
           (9.92261,-62.27056), (9.90253,-62.35468), (10.00881,-62.41047),
           (10.21876,-62.53449), (10.20749,-62.58673), (10.1854,-62.64996),
           (10.40338,-62.69126), (10.53481,-62.8699), (10.587,-62.62963), 
           (10.5969,-62.35706), (10.70442,-61.87745), (10.70554,-62.29957),
           (10.73714,-62.58479), (10.72595,-62.7686), (10.62839,-63.55116),
           (10.63329,-63.79013), (10.64654,-64.06616), (10.66396,-64.23989),
           (10.58629,-64.23337), (10.60414,-64.09552), (10.56717,-63.82885),
           (10.50663,-63.64408), (10.44394,-63.74944), (10.445,-64.01087), 
           (10.46227,-64.11024), (10.41848,-64.27741), (10.23687,-64.4281),
           (10.24576,-64.54994), (10.04473,-64.83883), (10.06047,-64.98053), 
           (10.09539,-65.24639), (10.15881,-65.54686), (10.23376,-65.76301), 
           (10.47251,-66.10317), (10.59451,-66.11121), (10.6312,-66.20037), 
           (10.62525,-66.37953), (10.6355,-66.86631), (10.48877,-67.62849), 
           (10.4694,-68.0755), (10.7089,-68.31085), (10.93376,-68.26648), 
           (11.21235,-68.46655), (11.41624,-68.78841), (11.50468,-69.17731), 
           (11.48699,-69.586), (11.57771,-69.69146), (12.09039,-69.85186), 
           (12.17206,-69.90301), (12.20539,-69.99262), (12.0991,-70.15364), 
           (11.88204,-70.25063), (11.82506,-70.23299), (11.77344,-70.17141),
           (11.65457,-70.17575), (11.74306,-69.79614), (11.54521,-69.70658), 
           (11.4247,-69.74283), (11.46134,-69.85159), (11.51196,-69.93344), 
           (11.42422,-70.09247), (11.30224,-70.32853), (11.18465,-70.69925), 
           (11.04498,-71.06863), (10.93737,-71.29727), (10.82685,-71.41211), 
           (10.75808,-71.46953), (10.6731,-71.44455), (10.55978,-71.47699), 
           (10.15994,-71.23503), (9.74876,-70.96573), (9.53214,-71.03574), 
           (9.2937,-71.02335), (9.06052,-71.38174), (9.01441,-71.62137), 
           (9.81915,-72.11367), (10.24061,-71.87833), (10.42761,-71.72954), 
           (10.50082,-71.64004), (10.69548,-71.5835), (10.97794,-71.74508), 
           (11.03773,-71.7017), (11.11368,-71.78467), (11.29467,-71.91432), 
           (11.57892,-71.95718), (11.68894,-71.77289), (11.74393,-71.52968), 
           (11.84401,-71.31194), (12.06062,-71.10695), (12.22523,-71.17611), 
           (12.31824,-71.2615), (12.42194,-71.54465), (12.30898,-71.82453), 
           (12.15316,-71.88619), (12.19339,-72.04786), (12.01343,-72.14361), 
           (11.87099,-72.2272), (11.75,-72.3767), (11.72847,-72.57641), 
           (11.62088,-72.77613), (11.29448,-73.19048), (11.21838,-73.7394), 
           (11.2864,-73.89008), (10.77182,-74.29363), (10.91513,-74.68619), 
           (10.70391,-75.17942), (10.25479,-75.46391), (9.62637,-75.52867), 
           (9.24665,-75.97853), (8.82311,-76.29655), (8.58612,-76.71424), 
           (8.22843,-76.67041), (7.95747,-76.79136), (8.48132,-77.32442), 
           (8.87326,-77.73376), (9.27562,-78.26396), (9.4608,-79.17875), 
           (9.49918,-79.55923), (9.0822,-80.53298), (8.71429,-81.23991), 
           (8.95426,-82.10064), (9.07474,-82.35163), (9.41203,-82.47078), 
           (9.83897,-83.12318), (10.62731,-83.64375), (11.39207,-84.03247), 
           (12.8449,-83.62079), (14.2255,-83.46179), (14.81294,-83.37924), 
           (15.21866,-83.79107), (15.7107,-84.37304), (15.88978,-85.03161), 
           (15.71936,-86.19494), (15.73394,-87.39122), (15.60044,-88.10411), 
           (15.81968,-88.92401), (16.10204,-88.97487), (16.5392,-88.56023), 
           (16.94384,-88.34334), (17.33713,-88.37914)]
    polygon = Polygon(coords)

    plt.figure()
    plt.plot(*polygon.exterior.xy)
    plt.scatter(*zip(*coords), s=15, c='goldenrod') 

    #retornamos el poligono
    return polygon

### Agrega columna de datos en poligono ###
def in_polygon (data, polygon):
    points = list(data[["latitude", "longitude"]].to_records(index=False))
    data_in_polygon = []
    i = 0
    print("Agregando columna si esta en poligono")
    
    for index, row in data.iterrows():
        if(Point(points[i]).within(polygon) == True):
            ### Agregar si esta o no en el poligono ###
            data_in_polygon.append(True)
        else:
            data_in_polygon.append(False)
        #print(row)
        i = i + 1
        
    data["in_polygon"] = data_in_polygon
    
    data2 = data[data.in_polygon == True]
    return data2        

### obteniendo las ultimas coords de los barcos ###
def get_last_position_mmsi(data_in_polygon):
    data_in_polygon = data_in_polygon.sort_values('timestamp').groupby('mmsi').tail(1)
    return data_in_polygon

### Creando coords de mmsi en grafica ###
def create_coords_mmsi(data_in_polygon):
    mmsi = data_in_polygon.mmsi.unique()
    i = 0
    for x in mmsi:
        ### Generamos color ramdon ###
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)


        data_for_mmsi = data_in_polygon.loc[data_in_polygon["mmsi"] == x]
        points = list(data_for_mmsi[["latitude", "longitude"]].to_records(index=False))
        plt.scatter(*zip(*points), s=15, c=color, label='tab:{}'.format(x), edgecolors='none')
        #plt.scatter(*zip(*points), s=15, c=color, label='tab:{}'.format(x), edgecolors='none') ## Con label
        
        if(i == 10):
            break
        i = i+1
        print(i)

### Agregando grid ###
def add_grid(tamaño):
    x_ticks = np.arange(7, 23, tamaño)
    y_ticks = np.arange(-59,-91, -(tamaño))
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(which='major')



### Guardando datos que estan dentro de poligono ###
def save_csv_within_polygon(data_in_polygon):
    data_in_polygon.to_csv("./data_in_polygon.csv")
    




#############################################################
### Copy
"""
### Creando las rutas de los mmsi en grafica ###
def create_coords_mmsi(data_in_polygon):
    points = list(data_in_polygon[["latitude", "longitude"]].to_records(index=False))
    #data = Polygon(points)
    plt.scatter(*zip(*points), s=15, c='green')
"""
   
### Agrega columna de datos en poligono ###
"""
def delete_points_outside_polygon(data, polygon):
    points = list(data[["latitude", "longitude"]].to_records(index=False))
    data_in_polygon = []
    #data_in_polygon = data
    i = 0
    for index, row in data.iterrows():
        
        if(Point(points[i]).within(polygon) == True):
            ### agg los puntos que estan dentro del poligono ###
            data_in_polygon.append(1)
        else:
            data_in_polygon.append(0)
            ### Borrar los puntos que estan fuera del poligono ###
            #data_in_polygon = data_in_polygon.drop(index)
        #print(row)
        i = i+1
        print(i)
    return data_in_polygon
"""

    
    
