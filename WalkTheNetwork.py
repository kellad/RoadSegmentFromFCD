#!/usr/bin/env python
# coding: utf-8

import math

grid = {}
sourceFile = "C:/SegmentGeneration/202112_c10_b_rwr.txt"
targetFile = "C:/SegmentGeneration/202112_c10_b_or.txt"
vertexFile = "C:/SegmentGeneration/202112_c10_b_vertex.txt"
multiplier = 100000
gridWidth = 10
shifter =  math.pow(10, math.log10(multiplier) + 3)
with open(sourceFile, "r") as r:
        while True:
            line = r.readline()
            if not line:
                break
            cells = line.split(';')
            if cells[0] == '' or cells[1] == "dist2road":
                continue
            gridCells = []
            for cell in cells:
                gridCells.append(float(cell))
            #(long)((Math.Floor((latitude+90)*multiplier / gridWidth) * gridWidth) * Math.Pow(10, Math.Log10(multiplier) + 3) + (Math.Floor((longitude + 180)*multiplier / gridWidth) * gridWidth));
            gridKey = math.floor((gridCells[2] + 90) * multiplier / gridWidth) * gridWidth * shifter                       + math.floor((gridCells[3] + 180) * multiplier / gridWidth) * gridWidth
            grid[gridKey] = gridCells

#grid[1307724021222250]


maxDistance2Road = 0.00015
gridVisited = {}
gridOnRoad = {}
for l in range(9):        #Visit points and mark on-road or off-road
    for cellKey in grid:

        if cellKey in gridVisited: #Skip if already visited
            continue
            
        #gridOnRoad[cellKey] = False #Assume point is offroad
        
        if grid[cellKey][1] > maxDistance2Road: #skip if distance to nearest road is more than max allowed distance        
            gridVisited[cellKey] = True #Point visited
            continue
        
        n = cellKey + gridWidth * shifter #North neighbour key
        ne = n + gridWidth                #Northeast neighbour key
        e = cellKey + gridWidth           #East neighbour key
        se = e - gridWidth * shifter      #Southeast neighbour key
        s = cellKey - gridWidth * shifter #South neighbour key
        sw = s - gridWidth                #Southwest neighbour key
        w = cellKey - gridWidth           #West neighbour key
        nw = w + gridWidth * shifter      #Northwest neighbour key
        
        neighbours = [n,ne,e,se,s,sw,w,nw]
        pointCounts = [0,0,0,0,0,0,0,0]    
        
        neighbourCountOnRoad = 0
        sameRoadNeighboursMaxPointCount = 0
        uniqueSegments = {} 
        maxCountNeighbour = -1

        for i in range(8):
            neighbour = neighbours[i]            
            if neighbour in grid:
                uniqueSegments[grid[neighbour][0]] = 1 #For counting unique segments in the 8-neighbourhood
                if not neighbour in gridOnRoad: # Skip if cell is on some road
                    if grid[neighbour][1] > maxDistance2Road:
                        continue
                    pointCounts[i] = grid[neighbour][4] #if cell exists get point count        
                    if grid[cellKey][0] == grid[neighbour][0] and sameRoadNeighboursMaxPointCount < pointCounts[i]: # if pointcount higher than current max
                        sameRoadNeighboursMaxPointCount = pointCounts[i]
                        maxCountNeighbour = neighbour
                elif grid[cellKey][0] == grid[neighbour][0]:
                    neighbourCountOnRoad += 1 # Add to count if this neighbour is on same road

        if cellKey == 1307373021219420 or cellKey == 1307373021219430 or cellKey == 1307372021219430:
            print(cellKey)
            print('\n')

        pointLimit = 2 + len(uniqueSegments)
        #if len(uniqueSegments) > 1:
        #    pointLimit = 2

        if neighbourCountOnRoad < pointLimit:    
            if grid[cellKey][4] >= sameRoadNeighboursMaxPointCount: 
                gridOnRoad[cellKey] = True
                gridVisited[cellKey] = True #Point visited
            elif grid[maxCountNeighbour][1] < maxDistance2Road and neighbourCountOnRoad < 2: #if this is not max and there is no neighbour on road mark max count neighbour as on-road
                gridOnRoad[maxCountNeighbour] = True
                gridVisited[maxCountNeighbour] = True #Point visited
        else:
            gridVisited[cellKey] = True #Point visited

from shapely.geometry import Point

for cellKey in gridOnRoad:
    
    if not gridOnRoad[cellKey]:
        continue

    n = cellKey + gridWidth * shifter #North neighbour key
    ne = n + gridWidth                #Northeast neighbour key
    e = cellKey + gridWidth           #East neighbour key
    se = e - gridWidth * shifter      #Southeast neighbour key
    s = cellKey - gridWidth * shifter #South neighbour key
    sw = s - gridWidth                #Southwest neighbour key
    w = cellKey - gridWidth           #West neighbour key
    nw = w + gridWidth * shifter      #Northwest neighbour key
    
    neighbours = [n,ne,e,se,s,sw,w,nw]

    for i in range(8): #For each neighbour
        neighbour = neighbours[i]            
        if neighbour in gridOnRoad: #if neighbour marked as on-road
            #if grid[cellKey][0] == grid[neighbour][0]:  #if neighbour is closest to same road
            currentPoint = Point(grid[cellKey][3], grid[cellKey][2]) 
            neighbourPoint = Point(grid[neighbour][3], grid[neighbour][2])
            interPointsDistance = currentPoint.distance(neighbourPoint)
            if grid[cellKey][1] > ((math.sqrt(2) * gridWidth / multiplier) / 2 + grid[neighbour][1]): #if distance to road bigger than distance to this neighbour mark as off-road
                gridOnRoad[cellKey] = False

for cellKey in grid:
    if cellKey in gridOnRoad:
        if gridOnRoad[cellKey]:
            continue

    n = cellKey + gridWidth * shifter #North neighbour key
    ne = n + gridWidth                #Northeast neighbour key
    e = cellKey + gridWidth           #East neighbour key
    se = e - gridWidth * shifter      #Southeast neighbour key
    s = cellKey - gridWidth * shifter #South neighbour key
    sw = s - gridWidth                #Southwest neighbour key
    w = cellKey - gridWidth           #West neighbour key
    nw = w + gridWidth * shifter      #Northwest neighbour key
    
    neighbours = [n,ne,e,se,s,sw,w,nw]
    neighbourRoads = [0,0,0,0,0,0,0,0]

    for i in range(8): #For each neighbour
        neighbour = neighbours[i]            
        if neighbour in grid:
            if neighbour in gridOnRoad:
                if gridOnRoad[neighbour]:
                    neighbourRoads[i] = grid[neighbour][0]

    if neighbours[0] in grid:
        if neighbourRoads[0] == 0 and ((grid[cellKey][0] == neighbourRoads[7] and grid[cellKey][0] == neighbourRoads[2]) or (grid[cellKey][0] == neighbourRoads[6] and grid[cellKey][0] == neighbourRoads[1])):
            if grid[cellKey][4] > grid[neighbours[0]][4]:    
                gridOnRoad[cellKey] = 1
            else:
                gridOnRoad[neighbours[0]] = 1

    if neighbours[2] in grid:
        if neighbourRoads[2] == 0 and ((grid[cellKey][0] == neighbourRoads[1] and grid[cellKey][0] == neighbourRoads[4]) or (grid[cellKey][0] == neighbourRoads[0] and grid[cellKey][0] == neighbourRoads[3])):
            if grid[cellKey][4] > grid[neighbours[2]][4]:    
                gridOnRoad[cellKey] = 1
            else:
                gridOnRoad[neighbours[2]] = 1

    if neighbourRoads[2] == 0 and neighbourRoads[6] == 0 and grid[cellKey][0] == neighbourRoads[0] and grid[cellKey][0] == neighbourRoads[4]:
        gridOnRoad[cellKey] = 1
    if neighbourRoads[3] == 0 and neighbourRoads[7] == 0 and grid[cellKey][0] == neighbourRoads[1] and grid[cellKey][0] == neighbourRoads[5]:
        gridOnRoad[cellKey] = 1
    if neighbourRoads[4] == 0 and neighbourRoads[0] == 0 and grid[cellKey][0] == neighbourRoads[2] and grid[cellKey][0] == neighbourRoads[6]:
        gridOnRoad[cellKey] = 1
    if neighbourRoads[5] == 0 and neighbourRoads[1] == 0 and grid[cellKey][0] == neighbourRoads[3] and grid[cellKey][0] == neighbourRoads[7]:
        gridOnRoad[cellKey] = 1

with open(targetFile, "w") as w:
    with open(sourceFile, "r") as r:
        while True:
            line = r.readline()
            if not line:
                break
            cells = line.split(';')
            if cells[0] == '' or cells[1] == "dist2road":
                continue

            gridKey = math.floor((float(cells[2]) + 90) * multiplier / gridWidth) * gridWidth * shifter \
                      + math.floor((float(cells[3]) + 180) * multiplier / gridWidth) * gridWidth

            onRoad = "0"
            if gridKey in gridOnRoad:
                if gridOnRoad[gridKey]:
                    onRoad = "1"
            w.write(onRoad)
            w.write(";")
            w.write(line.replace("\n",";" + str(gridKey) + "\n"))

with open(vertexFile, "w") as v:                
    with open(targetFile, "w") as w:
        with open(sourceFile, "r") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                cells = line.split(';')
                if cells[0] == '' or cells[1] == "dist2road":
                    continue

                gridKey = math.floor((float(cells[2]) + 90) * multiplier / gridWidth) * gridWidth * shifter \
                        + math.floor((float(cells[3]) + 180) * multiplier / gridWidth) * gridWidth

                onRoad = "0"
                if gridKey in gridOnRoad:
                    if gridOnRoad[gridKey]:
                        onRoad = "1"
                w.write(onRoad)
                w.write(";")
                w.write(line.replace("\n",";" + str(gridKey) + "\n"))
                    
    with open(targetFile, "r") as r:
        with open(vertexFile, "w") as v:
            while True:
                line = r.readline()
                if not line:
                    break
                cells = line.split(';')
                if cells[0] == '' or cells[1] == "dist2road":
                    continue

                gridKey = math.floor((float(cells[2]) + 90) * multiplier / gridWidth) * gridWidth * shifter \
                        + math.floor((float(cells[3]) + 180) * multiplier / gridWidth) * gridWidth

                onRoad = "0"
                if gridKey in gridOnRoad:
                    if gridOnRoad[gridKey]:
                        onRoad = "1"
                w.write(onRoad)
                w.write(";")
                w.write(line.replace("\n",";" + str(gridKey) + "\n"))
                    


