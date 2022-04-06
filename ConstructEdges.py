#Construct edges
import math
import sys

vertexFile = "C:/SegmentGeneration/202112_c10_y_or.txt"
edgeMap = "C:/SegmentGeneration/202112_c10_y_edges.txt"
edgeFile = "C:/SegmentGeneration/202112_c10_y_edges_data.txt"
vectorDataFile = "C:/SegmentGeneration/202112_c10_y_vector_data.txt"
vertexDataFile = "C:/SegmentGeneration/202112_c10_y_vertex.txt"
multiplier = 100000
gridWidth = 10
shifter =  math.pow(10, math.log10(multiplier) + 3)

vertexDict = {}
indexVertexDict = {}
stepLimit = 4
edgeDict = {}

def addEdge(vertex1,vertex2):
    if not vertex1 in edgeDict:                #if vertex1 is not present in edge dictionary
        edgeDict[vertex1] = {vertex2:None}     #add vertex1 and neigbor to edge dictionary
                    
    if not vertex2 in edgeDict:                #same for other direction
        edgeDict[vertex2] = {vertex1:None}
    
    if not vertex1 in edgeDict[vertex2]:       #if vertex1 is present but neighbor is not in edge dictionary
        edgeDict[vertex2][vertex1] = None      #add edge with neighbor

    if not vertex2 in edgeDict[vertex1]:       #same for other direction
        edgeDict[vertex1][vertex2] = None    

    onRoad = '0'
    if vertexDict[vertex1][0]=='1' and vertexDict[vertex2][0]=='1':
        onRoad = '1'

    w.write(str(vertex1))
    w.write(";")
    w.write(str(vertex2))
    w.write(";")
    w.write(onRoad)
    w.write(";LINESTRING(")
    w.write(str(vertexDict[vertex1][4]))
    w.write(" ")
    w.write(str(vertexDict[vertex1][3]))
    w.write(",")
    w.write(str(vertexDict[vertex2][4]))
    w.write(" ")
    w.write(str(vertexDict[vertex2][3]))
    w.write(")")
    # for vertex in [vertex1,vertex2]:
    #     for i in range(len(vertexDict[vertex])):
    #         w.write(";")
    #         w.write(vertexDict[vertex][i])
    w.write("\n")
    
#Load vertex dictionary
with open(vertexFile, "r") as r:
    while True:
        line = r.readline()
        if not line:
            break
        
        cells = line.split(";")

        if len(cells) < 33:
            continue

        gridKey = math.floor((float(cells[3]) + 90) * multiplier / gridWidth) * gridWidth * shifter \
                      + math.floor((float(cells[4]) + 180) * multiplier / gridWidth) * gridWidth
        
        vertexDict[gridKey] = cells
        indexVertexDict[gridKey] = cells[32]

#Connect edges
with open(edgeMap, "w") as w:
    for vertex in vertexDict:
        x = vertex % shifter
        y = vertex // shifter

        #Connect north
        yStep = 0
        xStep = 0
        while yStep < stepLimit:
            north = (y + gridWidth * (1+yStep)) * shifter + x + xStep * gridWidth
            if north in vertexDict:
                addEdge(vertex, north)
                # w.write(str(vertex))
                # w.write(";")
                # w.write(str(north))
                # w.write(";")
                # w.write(str(vertexDict[vertex][3]))
                # w.write(";")
                # w.write(str(vertexDict[vertex][4]))
                # w.write(";")
                # w.write(str(vertexDict[north][3]))
                # w.write(";")
                # w.write(str(vertexDict[north][4]))
                # w.write("\n")
                
                break
            else:
                if xStep < yStep:
                    xStep += 1
                    continue
                else:
                    yStep += 1
                    xStep = 0
                    continue 

        #Connect northeast
        yStep = 0
        xStep = 0
        while yStep < stepLimit:
            northeast = (y + gridWidth * (1+yStep)) * shifter + x + (1 + xStep) * gridWidth 
            if northeast in vertexDict:
                addEdge(vertex, northeast)
                break
            else:
                if yStep < xStep:
                    yStep += 1
                    continue
                else:
                    xStep += 1
                    yStep = 0
                    continue 
        #Connect east
        yStep = 0
        xStep = 0
        while yStep < stepLimit:
            east = (y + gridWidth * (-yStep)) * shifter + x + (1 + xStep) * gridWidth 
            if east in vertexDict:
                addEdge(vertex, east)
                break
            else:
                if yStep < xStep:
                    yStep += 1
                    continue
                else:
                    xStep += 1
                    yStep = 0
                    continue 

        #Connect southeast
        yStep = 0
        xStep = 0
        while yStep < stepLimit:
            southeast = (y + gridWidth * (-1 - yStep)) * shifter + x + (1 + xStep) * gridWidth
            if southeast in vertexDict:
                addEdge(vertex, southeast)
                break
            else:
                if xStep < yStep:
                    xStep += 1
                    continue
                else:
                    yStep += 1
                    xStep = 0
                    continue 
    
    #Clean duplicate edges and construct undirected graph
    edgeDictUndirected = {}
    for vertex1 in edgeDict:
        for vertex2 in edgeDict[vertex1]:
            if vertex2 in edgeDictUndirected:
                if vertex1 in edgeDictUndirected[vertex2]:
                    continue
            if not vertex1 in edgeDictUndirected: 
                edgeDictUndirected[vertex1] = {vertex2:None}
            else:
                edgeDictUndirected[vertex1][vertex2] = None

    with open(edgeFile, "w") as w:
        w.write("source,target,direction,length\n")
        for vertex1 in edgeDictUndirected:
            for vertex2 in edgeDictUndirected[vertex1]:
                w.write(str(int(indexVertexDict[vertex1])))
                w.write(",")
                w.write(str(int(indexVertexDict[vertex2])))
                w.write(",")

                direction = math.atan((float(vertexDict[vertex1][3]) - float(vertexDict[vertex2][3])) / (float(vertexDict[vertex1][4]) - float(vertexDict[vertex2][4]) + sys.float_info.epsilon))            
                if direction < 0:
                    direction += math.pi/2
                if direction > (math.pi/2):
                    direction -= math.pi/2
                direction /= (math.pi/2) 

                w.write("{:.10f}".format(direction))
                w.write(",")
                
                length = (multiplier / gridWidth**2) * math.sqrt((float(vertexDict[vertex1][3]) - float(vertexDict[vertex2][3]))**2 + (float(vertexDict[vertex1][4]) - float(vertexDict[vertex2][4]))**2)
                
                w.write("{:.10f}".format(length))
                w.write("\n")
                
                edgeDict[vertex1][vertex2] = [direction,length]
                edgeDict[vertex2][vertex1] = [direction,length]

### Reading vertex data into dictionary
vertexDataDict = {}
with open(vertexDataFile, "r") as r:
    while True:
        line = r.readline()
        if not line:
            break
        
        cells = line.split(",")

        if len(cells) < 30:
            continue
        
        if cells[0] == "index":
            continue

        vertexDataDict[int(cells[0])] = line
    
    
### Writing vector data file
with open(vectorDataFile, "w") as w:
    for cellKey in vertexDict:
        x = cellKey % shifter
        y = cellKey // shifter
        
        neighbors = []
        
        n  = cellKey + gridWidth * shifter              #North neighbour key
        neighbors.append(n)
        ne = cellKey + gridWidth * shifter + gridWidth  #Northeast neighbour key
        neighbors.append(ne)
        e  = cellKey                       + gridWidth  #East neighbour key
        neighbors.append(e)
        se = cellKey - gridWidth * shifter + gridWidth  #Southeast neighbour key
        neighbors.append(se)
        s  = cellKey - gridWidth * shifter              #South neighbour key
        neighbors.append(s)
        sw = cellKey - gridWidth * shifter - gridWidth  #Southwest neighbour key
        neighbors.append(sw)
        west  = cellKey                       - gridWidth  #West neighbour key
        neighbors.append(west)
        nw = cellKey + gridWidth * shifter - gridWidth  #Northwest neighbour key
        neighbors.append(nw)
        
        n2   = cellKey + 2 * gridWidth * shifter
        neighbors.append(n2)
        n2e  = cellKey + 2 * gridWidth * shifter +     gridWidth
        neighbors.append(n2e)
        n2e2 = cellKey + 2 * gridWidth * shifter + 2 * gridWidth
        neighbors.append(n2e2)
        ne2  = cellKey +     gridWidth * shifter + 2 * gridWidth
        neighbors.append(ne2)
        e2   = cellKey                           + 2 * gridWidth
        neighbors.append(e2)
        se2  = cellKey -     gridWidth * shifter + 2 * gridWidth
        neighbors.append(se2)
        s2e2 = cellKey - 2 * gridWidth * shifter + 2 * gridWidth
        neighbors.append(s2e2)
        s2e  = cellKey - 2 * gridWidth * shifter +     gridWidth
        neighbors.append(s2e)
        s2   = cellKey - 2 * gridWidth * shifter
        neighbors.append(s2)
        s2w  = cellKey - 2 * gridWidth * shifter -     gridWidth
        neighbors.append(s2w)
        s2w2 = cellKey - 2 * gridWidth * shifter - 2 * gridWidth
        neighbors.append(s2w2)
        sw2  = cellKey -     gridWidth * shifter - 2 * gridWidth
        neighbors.append(sw2)
        w2   = cellKey                           - 2 * gridWidth
        neighbors.append(w2)
        nw2  = cellKey +     gridWidth * shifter - 2 * gridWidth
        neighbors.append(nw2)
        n2w2 = cellKey + 2 * gridWidth * shifter - 2 * gridWidth
        neighbors.append(n2w2)
        n2w  = cellKey + 2 * gridWidth * shifter -     gridWidth
        neighbors.append(n2w)
        
        vertexDataKey = int(vertexDict[cellKey][32])
        w.write(vertexDataDict[vertexDataKey].replace('\n','').replace('onRoad','1').replace('offRoad','0').replace('dummy','0'))
        for i in range(24):
            if cellKey in edgeDict and neighbors[i] in vertexDict:                
                neighborDataKey = int(vertexDict[neighbors[i]][32])
                if neighborDataKey in vertexDataDict:
                    if neighbors[i] in edgeDict[cellKey]:
                        w.write(",")
                        w.write(str(edgeDict[cellKey][neighbors[i]][0]))
                        w.write(",")
                        w.write(str(edgeDict[cellKey][neighbors[i]][1]))
                        w.write(",")
                        startIndex = vertexDataDict[neighborDataKey].index(',', vertexDataDict[neighborDataKey].index(',') + 1 ) + 1
                        w.write(vertexDataDict[neighborDataKey][startIndex:].replace('\n',''))
                        continue
                    elif neighbors[i] in edgeDict:
                        w.write(",")
                        y1 = float(vertexDict[cellKey][3])
                        y2 = float(vertexDict[neighbors[i]][3])
                        x1 = float(vertexDict[cellKey][4])
                        x2 = float(vertexDict[neighbors[i]][4])                    
                        direction = math.atan((y1 - y2) / (x1 - x2 + sys.float_info.epsilon))            
                        if direction < 0:
                            direction += math.pi/2
                        if direction > (math.pi/2):
                            direction -= math.pi/2
                        direction /= (math.pi/2) 
        
                        w.write("{:.10f}".format(direction))
                        w.write(",")
                        
                        length = (multiplier / gridWidth**2) * math.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                        
                        w.write("{:.10f}".format(length))
                        
                        w.write(",")
                        startIndex = vertexDataDict[neighborDataKey].index(',', vertexDataDict[neighborDataKey].index(',') + 1 ) + 1
                        w.write(vertexDataDict[neighborDataKey][startIndex:].replace('\n',''))
                        continue

            w.write(",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
                    
        w.write("\n")
        
print("Process finished!")