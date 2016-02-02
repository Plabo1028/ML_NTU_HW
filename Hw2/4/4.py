import numpy as np
import pylab as pl
import math

def Original(N):
    return pow( (8/float(N) * math.log((4* pow(float(2*float(N)),50) / 0.05))), 0.5)

def Variant(N):
    return pow( (16/float(N) * math.log((2* pow(N,50) / pow(0.05,0.5) ))), 0.5)

def Rademacher(N):
    first_term = pow( (2*math.log(2*pow(N,51))/float(N)), 0.5)
    second_term = pow( (2/float(N)*math.log(1/0.05)), 0.5)
    return first_term+second_term+1/float(N)

def Parrondo(N):
    first_term = (4*math.log( (6* pow((2*N),50) / 0.05) ) / float(N))
    second_term = pow( (pow((2/float(N)),2) + first_term ),0.5)
    ansplus = (2/float(N) + second_term)/2
    ansminus = (2/float(N) - second_term)/2
    return [ansplus ,ansminus]

def Devroye(N):
    first_term = (4*(N-2)/float(N)) * (1/float(N)) * math.log((2* pow(N,50)/0.05))
    ansplus = (2 / float(N)) + first_term
    ansminus = (2 / float(N)) - first_term
    mother = ((2*N - 4) / float(N))
    return [(ansplus/mother),(ansminus/mother)]

Record_Original=[]
Record_Variant=[]
Record_Rademacher=[]
Record_Parrondo_plus=[]
#Record_Parrondo_minus=[]
Record_Devroye_plus=[]
#Record_Devroye_minus=[]
NumberList=range(5,11000,50)

for i in NumberList:
    Record_Original.append(Original(i))
    Record_Variant.append(Variant(i))
    Record_Rademacher.append(Rademacher(i))

    ansplus,ansminus=Parrondo(i)
    Record_Parrondo_plus.append(ansplus)
    #Record_Parrondo_minus.append(ansminus)

    ansplus,ansminus=Devroye(i)
    Record_Devroye_plus.append(ansplus)
    #Record_Devroye_minus.append(ansminus)

print 'N=10000:'
print '\tNormal: {0}'.format(Original(10000))
print '\tVariant: {0}'.format(Variant(10000))
print '\tRademacher: {0}'.format(Rademacher(10000))
print '\tParrondo_plus: {0}'.format(Parrondo(10000)[0])
print '\tDevroye_plus: {0}'.format(Devroye(10000)[0])
print '============================================'
print 'N=5:'
print '\tNormal: {0}'.format(Original(5))
print '\tVariant: {0}'.format(Variant(5))
print '\tRademacher: {0}'.format(Rademacher(5))
print '\tParrondo_plus: {0}'.format(Parrondo(5)[0])
print '\tDevroye_plus: {0}'.format(Devroye(5)[0])

pl.plot(NumberList, Record_Original, 'ro',label='Original')
pl.plot(NumberList, Record_Variant, 'bx',label='Variant')
pl.plot(NumberList, Record_Rademacher, 'g^',label='Rademacher')
pl.plot(NumberList, Record_Parrondo_plus, 'ys',label='Parrondo_plus')
#pl.plot(NumberList, Record_Parrondo_minus, 'ys',label='Parrondo_minus')
pl.plot(NumberList, Record_Devroye_plus, 'c*',label='Devroye_plus')
#pl.plot(NumberList, Record_Devroye_minus, 'c*',label='Devroye_minus')
legend = pl.legend(loc='center right', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FFCC')
pl.show()



