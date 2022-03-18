
import math
import numpy

multiplier = 100000;    #Multiply lon,lat degree with this
gridWidth = 10;         #10*10 grids
binAngle = 15;          #360 / 15 = 24 bins 
shifter =  math.pow(10, math.log10(multiplier) + 3)

class CPoint:
    def __init__(self):
        self.count = 0
        self.speedAverage = 0.0
        self.angleBins = numpy.zeros((360//binAngle), dtype=int)
        self.longitude = 0.0
        self.latitude = 0.0
    
    def addPoint(self, latitude, longitude, speed, bearing):
        self.latitude = (self.latitude * self.count + latitude) / (self.count + 1)
        self.longitude = (self.longitude * self.count + longitude) / (self.count + 1)
        self.speedAverage = (self.speedAverage * self.count + speed) / (self.count + 1)        
        self.angleBins[int(math.floor(bearing/binAngle))] += 1
        self.count += 1               

combinedFilePath = "C:\\SegmentGeneration\\202112.txt"
compressedFilePath =  "C:\\SegmentGeneration\\202112_c10_b.txt"
pointDict = {}

print("Building the dictionary...")
with open(combinedFilePath, "r") as r:
    while True:
        line = r.readline()
        if not line:
            break
        latitude = 0.0
        longitude = 0.0
        speed = 0.0
        bearing = 0.0

        cells = line.split(";")
        if(len(cells) < 10):
            continue

        try:
            latitude = float(cells[4])
            longitude = float(cells[5])
            speed = float(cells[6])
            bearing = float(cells[7])
        except ValueError:
            continue

        gridKey = math.floor((float(latitude) + 90) * multiplier / gridWidth) * gridWidth * shifter \
                  + math.floor((float(longitude) + 180) * multiplier / gridWidth) * gridWidth

        if not (gridKey in pointDict):
            pointDict[gridKey] = CPoint()

        try:
            pointDict[gridKey].addPoint(latitude,longitude,speed,bearing)
        except IndexError:
            continue

print("Saving to the file...")
with open(compressedFilePath, "w") as w:
    for key in pointDict:
        w.write(str(pointDict[key].latitude))
        w.write(";")
        w.write(str(pointDict[key].longitude))
        w.write(";")
        w.write(str(pointDict[key].count))
        w.write(";")
        w.write(str(pointDict[key].speedAverage))
        for binCount in pointDict[key].angleBins:
            w.write(";")
            w.write(str(binCount))
        w.write("\n")

print("Process completed!")
        
        
        
        
