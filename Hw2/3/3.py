import math

left = 0.05 / pow(2.0,12)

def function(N):
    return pow(N,10) * math.exp(-(0.0025/8)*N)
#print left
Recordlist = []
for a in range(420000,500000,1000):
    tmp = [a,function(a)]
    Recordlist.append(tmp)
#print Recordlist

distance = []
for a in Recordlist:
    tmp = [a[0],abs(left-a[1])]
    distance.append(tmp)
minindex = 0
minvalue = 1

for a in distance:
    if a[1] < minvalue:
        minvalue = a[1]
        minindex = a[0]

#print minvalue
print minindex
