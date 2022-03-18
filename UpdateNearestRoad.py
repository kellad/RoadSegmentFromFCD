#!/usr/bin/env python
# coding: utf-8

import geopandas
from shapely.geometry import Point

df = geopandas.read_file("C:/SegmentGeneration/NetworkGerede4.geojson")  


#df.head()

#p = Point(32.196661533267104, 40.74364318570015)

#gen = df.sindex.nearest(p)

#gen

l = list(df.items())

#l[1][1][gen[1][0]]

#l[0][1][gen[1][0]]

#"{:.6f}".format(l[1][1][gen[1][0]].distance(p))

sourceFile = "C:/SegmentGeneration/202112_c10_b.txt"
targetFile = "C:/SegmentGeneration/202112_c10_b_rwr.txt" #Related With Road
with open(targetFile, "w") as w:
    with open(sourceFile, "r") as r:
        while True:
            line = r.readline()
            if not line:
                break
            cells = line.split(';')
            if cells[0] == '' or cells[1] == "lon":
                continue
            lat = float(cells[0])
            lon = float(cells[1])
            p = Point(lon,lat)
            gen = df.sindex.nearest(p)
            w.write(str(int(l[0][1][gen[1][0]])))
            w.write(";")
            w.write("{:.7f}".format(l[3][1][gen[1][0]].distance(p)))
            w.write(";")
            w.write(line)              
