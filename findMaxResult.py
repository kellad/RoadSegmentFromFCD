# find max result
# Engin Yegnidemir 
# 8 March 2022

testCases = [
             [[3,4,5,1],72],
             [[1,1,1,5],15],
             [[1,0,2,3],9],
             [[-1,-2,-3,-4,2],72],
             [[1,0.5,0.6,0.9,10,-2],60],
             [[0.5,0.5,0.5],1.5],             
             [[1,1],2],
             [[-1,-1],2],
             [[3,3],9],
             [[3,1.5,3],13.5],
             [[3,1.5,2.5],12],
             [[3,0,1.5,0,2.5],12],
             [[0],0]
            ]

def findMaxResult(arr):

    if len(arr)  < 1:
        raise TypeError
    elif len(arr) < 2:
        return abs(arr[0])
    elif len(arr) > 2:
        i = 1
        while i < len(arr)-1: #look at a window of three numbers
            current = abs(arr[i])
            left = abs(arr[i-1])
            right = abs(arr[i+1])
            if current < 2:
                if left < right:
                    if current * (left - 2) / (2 - current) < left:
                        add = left+current
                        arr.pop(i-1)
                        arr.pop(i-1)
                        arr.insert(i-1,add)
                        continue
                else:
                    if current * (right - 2) / (2 - current) < right:
                        add = right+current
                        arr.pop(i)
                        arr.pop(i)
                        arr.insert(i,add)
                        continue
            i += 1 #if there is nothing to add advance to next window
    if len(arr) > 1:        
        current = abs(arr[0])
        if current < 2:
            right = abs(arr[1])
            if current * (right - 2) / (2 - current) < right :
                add = current+right
                arr.pop(0)
                arr.pop(0)
                arr.insert(0,add)
               

    if len(arr) > 1:
        lastIndex = len(arr) - 1
        current = abs(arr[lastIndex])        
        if current < 2:
            left = abs(arr[lastIndex - 1])
            if current  * (left - 2) / (2 - current) < left:
                add = arr[lastIndex - 1]+current
                arr.pop(lastIndex - 1)
                arr.pop(lastIndex - 1)
                arr.insert(lastIndex - 1,add)               

    res = arr[0]
    for i in range(1,len(arr)):
        res = abs(arr[i] * res)

    return res

for testCase in testCases:
    #testCase = testCases[6]
    print(str(findMaxResult(testCase[0])) + '=' + str(testCase[1]))
        