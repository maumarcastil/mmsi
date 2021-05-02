import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import functions as fn


path = "Universidad_del_Norte_Columbian_EEZ_000000000002.csv"
data = fn.read_and_clean_csv(path)


### Creando Poligono en grafica###
polygon = fn.create_polygon()


### Dataframe nuevo con los datos dentro y fuera   ###
data_in_polygon = fn.in_polygon(data, polygon)


### Guardando datos que estan dentro de poligono ###
#fn.save_csv_within_polygon(data_in_polygon)


### Leyendo csv con los puntos que estan dentro del poligono ###
#data2 = fn.read_csv()


### Creando las rutas de los mmsi en grafica ###
fn.create_coords_mmsi(data_in_polygon)


### Creando las ultimas coords de los barcos ###
#points_last_position = fn.get_last_position_mmsi(data_in_polygon) 
#fn.create_coords_mmsi(points_last_position)


### Mostrando Figura ###
#plt.legend(loc="upper right", title="mmsi")
fn.add_grid(2)
plt.show()

    
    
    
    
    
    
    
