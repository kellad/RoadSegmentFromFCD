#Construct edges
import math

vertexFile = "C:/SegmentGeneration/202112_c10_b_or.txt"
edgeFile = "C:/SegmentGeneration/202112_c10_b_edges.txt"
multiplier = 100000
gridWidth = 10
shifter =  math.pow(10, math.log10(multiplier) + 3)

vertexDict = {}
stepLimit = 4
edgeDict = {}

def addEdge(vertex1,vertex2):
    if vertex1 not in edgeDict:                #if vertex1 is not present in edge dictionary
        edgeDict[vertex1] = {vertex2:None}     #add vertex1 and neigbor to edge dictionary
                    
    if vertex2 not in edgeDict:                #same for other direction
        edgeDict[vertex2] = {vertex1:None}
    
    if vertex1 not in edgeDict[vertex2]:       #if vertex1 is present but neighbor is not in edge dictionary
        edgeDict[vertex2][vertex1] = None      #add edge with neighbor

    if vertex2 not in edgeDict[vertex1]:       #same for other direction
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

        if len(cells) < 31:
            continue

        gridKey = math.floor((float(cells[3]) + 90) * multiplier / gridWidth) * gridWidth * shifter \
                      + math.floor((float(cells[4]) + 180) * multiplier / gridWidth) * gridWidth
        
        vertexDict[gridKey] = cells

#Connect edges
with open(edgeFile, "w") as w:
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


print("Process finished!")