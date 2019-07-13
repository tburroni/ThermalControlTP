'''
Final project for Thermal Control
Astronautical Engineering, National University of San Martín
Professor: CASTELLO, Nahuel.
Students: BURRONI, Tomás Ignacio; ESCOBAR, Matías Ignacio.

2019/06/26
Version 2.0
Editor: Tom
'''



#LIBRARIES---------------------------------------------
# Libraries added as needed, avoid useless libraries


import numpy as np
import matplotlib.pyplot as plt



#CONSTANTS---------------------------------------------


kelvin = 273.15
qs = 2612.94                        # solar flux in W/m**2
sunlim1 = 65.5                      # true longitude angle from which the sun is seen
sunlim2 = 294.5                     # true longitude angle at which the sun is no longer seen
qp = 160                            # planet flux in W/m**2
alb = 0.76                          # Venus's albedo
alblim1 = 114.5                     # true longitude angle from which the planet is seen fully lit
alblim2 = 245.5                     # true longitude angle at which the planet is no longer seen fully lit
albran = alblim1 - sunlim1          # angle of the arc that covers the albedo transition
orbit = 2*np.pi * np.sqrt((6051.8+600)**3 / 3.24859e+5) # duration of the orbit in seconds
dt = orbit/1000                     # delta time in seconds, almost 6 seconds
Dt = orbit/30                       # big delta for initial transient, almost 200 sec
kAl = 130                           # thermal conductivity of aluminum in W/(mK)
AdL = .0003/.3                     # A/L of each contact in m**2/m
C = kAl*AdL                         # conductance between each node in W/K
mcE = 242.784                       # mass * heat capacity of each external node in J/K
mcI = 10000                         # mass * heat capacity of the internal node in J/K
area = 0.09                         # exposed area of the external nodes in m**2
sigma = 5.670374419e-8              # stefan boltzmann constant in W/m**2K**4
planetangle = 65.5*2*np.pi/360      # view cone half angle
planetSide = (planetangle - np.cos(planetangle)*np.sin(planetangle)) / np.pi    # view factor for the faces parallel to the planet vector
planetFront = (np.sin(planetangle))**2
SolarEff = 0.2                      # efficiency of the solar cells
CellArea = 1                      # fraction of face covered in solar cells (the rest is covered with white paint)
WhiteAlpha = 0.06                   # alpha of the paint used on the cells faces
WhiteEpsilon = 0.88                 # epsilon of the paint used on the cells faces
CoverAlpha = 0.06                   # alpha of the paint used on the Y faces
CoverEpsilon = 0.88                 # epsilon of the paint used on the Y faces
CellAlpha = 0.9
CellEpsilon = 0.83



#FUNCTIONS---------------------------------------------


# get true longitude from time ###
def t2trlong(t):
    trlong = 360 * t / orbit
    while trlong >= 360:
        trlong -= 360
    while trlong < 0:
        trlong += 360
    return trlong


# heat flux from the sun ###
def Qs(trlong, alpha, f, A):
    if trlong < sunlim1 or trlong > sunlim2:
        return 0
    else:
        return alpha*f*A*qs*(1-SolarEff)


# infrared heat flux ###
def Qir(epsilon, f, A):
    return epsilon*f*A*qp


# albedo heat flux ###
def Qa(trlong, alpha, f, A):
    if trlong <= sunlim1 or trlong >= sunlim2:
        lit = 0
    elif trlong >= alblim1 and trlong <= alblim2:
        lit = 1
    else:
        if trlong < alblim1:
            beta = (trlong - sunlim1) / albran
        else:
            beta = (sunlim2 - trlong) / albran
        lit = ((2*beta-1)*np.sqrt(1-(1-2*beta)**2) + np.arccos(1-2*beta)) / np.pi
    var = (trlong-65.5)/(294.5-65.5)*np.pi
    extra = np.sin(var)
    qa = alb * qs * lit * extra
    return alpha*f*A*qa*(1-SolarEff)


# heat radiated ###
def Qr(T,area,epsilon):
    #if True:
        #return 0                    # only used for verif
    return area*sigma*epsilon*(T**4)


# sun view factor ###
def SunF(trlong,angle):
    #if True:                        # only used for albedo verification
        #return 0
    if angle < 0:
        return 0
    dif = trlong-angle
    while dif >= 360:
        dif -= 360
    while dif < 0:
        dif += 360
    if dif >= 90 and dif <= 270:
        return 0
    else:
        return np.cos(dif*2*np.pi/360)


# total radiationflux ###
def Rad(node,trlong):
    return Qs(trlong,node.alpha,SunF(trlong,node.angle),node.areaexp) + Qir(node.epsilon,node.planetF,node.areaexp) + Qa(trlong,node.alpha,node.planetF,node.areaexp) - Qr(node.T,node.areaexp,node.epsilon)


# new temperatures ###
def NextTemp(sat,trlong,dt):
    # node 0
    Qcond = 2*C*(-4*sat[0].T + sat[1].T + sat[2].T + sat[5].T + sat[6].T)
    Qrad = Rad(sat[0],trlong)
    sat[0].Tnext = sat[0].T + (Qcond+Qrad)*dt/mcI
    # node 1
    Qcond = C*(-6*sat[1].T + 2*sat[0].T + sat[3].T + sat[4].T + sat[5].T + sat[6].T)
    Qrad = Rad(sat[1],trlong)
    sat[1].Tnext = sat[1].T + (Qcond+Qrad)*dt/mcE
    # node 2
    Qcond = C*(-6*sat[2].T + 2*sat[0].T + sat[3].T + sat[4].T + sat[5].T + sat[6].T)
    Qrad = Rad(sat[2],trlong)
    sat[2].Tnext = sat[2].T + (Qcond+Qrad)*dt/mcE
    # node 3
    Qcond = C*(-4*sat[3].T + sat[1].T + sat[2].T + sat[5].T + sat[6].T)
    Qrad = Rad(sat[3],trlong)
    sat[3].Tnext = sat[3].T + (Qcond+Qrad)*dt/mcI
    # node 4
    Qcond = C*(-4*sat[4].T + sat[1].T + sat[2].T + sat[5].T + sat[6].T)
    Qrad = Rad(sat[4],trlong)
    sat[4].Tnext = sat[4].T + (Qcond+Qrad)*dt/mcI
    # node 5
    Qcond = C*(-6*sat[5].T + 2*sat[0].T + sat[1].T + sat[2].T + sat[3].T + sat[4].T)
    Qrad = Rad(sat[5],trlong)
    sat[5].Tnext = sat[5].T + (Qcond+Qrad)*dt/mcE
    # node 6
    Qcond = C*(-6*sat[6].T + 2*sat[0].T + sat[1].T + sat[2].T + sat[3].T + sat[4].T)
    Qrad = Rad(sat[6],trlong)
    sat[6].Tnext = sat[6].T + (Qcond+Qrad)*dt/mcE
    return


def NextTempSimp(sat,trlong,dt):
    sat.Tnext = sat.T + Rad(sat,trlong)*dt/mcE
    return


def NextTempSimp2(sat,trlong,dt):
    # node 0
    Qcond = C*(-sat[0].T + sat[1].T)
    Qrad = Rad(sat[0],trlong)
    sat[0].Tnext = sat[0].T + (Qcond+Qrad)*dt/mcI
    # node 1
    Qcond = C*(-sat[1].T + sat[0].T)
    Qrad = Rad(sat[1],trlong)
    sat[1].Tnext = sat[1].T + (Qcond+Qrad)*dt/mcE
    return


# results plotter ###
def Plot(t,T,lim1,lim2,xl):
    fig1 = plt.figure(1, figsize=(16,12))
    plt.plot(t, T[:,0], 'b', label='Node 0 - \'Int\'', linewidth=2)
    plt.plot(t, T[:,1], 'g', label='Node 1 - \'+X\'', linewidth=2)
    plt.plot(t, T[:,2], 'r', label='Node 2 - \' -X\'', linewidth=2)
    plt.plot(t, T[:,3], 'c', label='Node 3 - \'+Z\'', linewidth=2)
    plt.plot(t, T[:,4], 'm', label='Node 4 - \' -Z\'', linewidth=2)
    plt.plot(t, T[:,5], 'y', label='Nodes 5,6 - \'+Y,-Y\'', linewidth=2)
    #plt.plot(t, T[:,6], 'k', label='Node 6 - \' -Y\'', linewidth=2)
    plt.xlabel(r'$\lambda_{tr}$ [$^o$]', fontsize=18)
    plt.ylabel(r'T [$^o$C]', fontsize=18)
    plt.title('Temperature during orbit',fontsize=30)
    plt.ylim([lim1,lim2])
    if xl:
        plt.xlim([0,360])
    plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
    plt.legend(loc='best')
    
    #fig1 = plt.figure(1, figsize=(16,12))
    #plt.plot(t, T[:,0], 'r', label='Node 1')
    #plt.plot(t, T[:,1], 'r', label='Node 2')
    #plt.plot(t, T[:,2], 'r', label='Node 3')
    #plt.plot(t, T[:,3], 'r', label='Node 4')
    #plt.plot(t, T[:,4], 'r', label='Node 5')
    #plt.plot(t, T[:,5], 'r', label='Node 6')
    #plt.plot(t, T[:,6], 'r', label='Node 7')
    #plt.xlabel('t [s]', fontsize=18)
    #plt.ylabel(r'T [$^o$C]', fontsize=18)
    #plt.title('Hot orbit',fontsize=30)
    #plt.ylim([-40,80])
    #plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
    #plt.legend(loc='upper right')
    
    #fig1 = plt.figure(1, figsize=(16,12))
    #plt.plot(t, T[:,7], 'b', label='Node 1')
    #plt.plot(t, T[:,8], 'b', label='Node 2')
    #plt.plot(t, T[:,9], 'b', label='Node 3')
    #plt.plot(t, T[:,10], 'b', label='Node 4')
    #plt.plot(t, T[:,11], 'b', label='Node 5')
    #plt.plot(t, T[:,12], 'b', label='Node 6')
    #plt.plot(t, T[:,13], 'b', label='Node 7')
    #plt.xlabel('t [s]', fontsize=18)
    #plt.ylabel(r'T [$^o$C]', fontsize=18)
    #plt.title('Cold orbit',fontsize=30)
    #plt.ylim([-40,80])
    #plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
    #plt.legend(loc='upper right')
    
    plt.show()
    return



#CLASSES---------------------------------------------


class Node:
    T = kelvin + 30                 # temperature at time t in degrees kelvin
    Tnext = kelvin + 30             # temperature at time t+dt
    mc = 0                          # mass * heat capacity in J/K
    Qg = 0                          # generated heat in W
    planetF = 0                     # view factor
    alpha = 0                       # solar absorption factor
    epsilon = 0                     # emissivity
    areaexp = 0                     # exposed area
    angle = 0                       # angle with respect to the sun at trlong = 0



#MAIN PROGRAM---------------------------------------------


print ('\n\n-------------------------------Thermal Control Simulator--------------------------------')
print ('Developed by:\n\tBURRONI, Tomas\n\tESCOBAR, Matias\n\n')


sat = [Node(),Node(),Node(),Node(),Node(),Node(),Node()]    # nodes int,+x,+y,+z,-x,-y,-z

sat[0].mc = mcI
sat[0].Qg = 50.0887*CellArea            # defined by the program, see calculations below

for i in range(1,7):
    sat[i].mc = mcE
    sat[i].areaexp = area
for i in range(1,5):
    sat[i].alpha = CellAlpha*CellArea + WhiteAlpha*(1-CellArea)
    sat[i].epsilon = CellEpsilon*CellArea + WhiteEpsilon*(1-CellArea)
for i in range(5,7):
    sat[i].alpha = CoverAlpha
    sat[i].epsilon = CoverEpsilon
for i in [1,2,5,6]:
    sat[i].planetF = planetSide
sat[4].planetF = 0
sat[3].planetF = planetFront

sat[1].angle = 90
sat[2].angle = 270
sat[3].angle = 0
sat[4].angle = 180
sat[5].angle = -1
sat[6].angle = -1

#sat2 = Node()
#sat2.mc = mcE
#sat2.planetF = 1
#sat2.alpha = 0.9
#sat2.epsilon = 0.83
#sat2.areaexp = area


#sat3 = [Node(),Node()]
#sat3[0].mc = mcE
#sat3[0].planetF = 1
#sat3[0].alpha = 0.9
#sat3[0].epsilon = 0.83
#sat3[0].areaexp = area
#sat3[1].mc = mcE
#sat3[1].planetF = 0
#sat3[1].alpha = 0.9
#sat3[1].epsilon = 0.83
#sat3[1].areaexp = area


## solar & albedo power calculation
#print ('Radiative influx')
#t = np.linspace(0,orbit,2*orbit+1)
#trl = np.zeros([len(t)])
#for i in range(1,7):
    #sat[i].epsilon = 0
#Q = np.zeros([len(t),6])
#for i in range(0,6):
    #for i2 in range(0,len(t)):
        #trl[i2] = t2trlong(t[i2])
        #Q[i2,i] = Rad(sat[i+1],trl[i2])
#E = Q.sum() * orbit / len(t) * SolarEff/(1-SolarEff)
#W = E / orbit
#print ('Total energy received = ', E, ', constant power = ', W)
###


### solar power calculation
#print ('Solar Flux')
#t = np.linspace(0,orbit,2*orbit+1)
#trl = np.zeros([len(t)])
#for i in range(1,5):
    #sat[i].epsilon = 0
    #sat[i].planetF = 0
#Q = np.zeros([len(t),4])
#for i in range(0,4):
    #for i2 in range(0,len(t)):
        #trl[i2] = t2trlong(t[i2])
        #Q[i2,i] = Rad(sat[i+1],trl[i2])
#Q2 = SolarEff /(1-SolarEff) * Q
#E = Q.sum() * orbit / len(t) * SolarEff/(1-SolarEff)
#W = E / orbit
#print ('Total energy received = ', E, ', constant power = ', W)
#fig1 = plt.figure(1, figsize=(16,12))
#plt.plot(trl, Q[:,0]+Q[:,1]+Q[:,2]+Q[:,3], 'b', label='Total Power', linewidth=2)
##plt.plot(trl, Q2[:,0]+Q2[:,1]+Q2[:,2]+Q2[:,3], 'b', label='Effective Power')
#plt.plot(trl, Q[:,0], 'r', label='Node 1', linewidth=2)
#plt.plot(trl, Q[:,1], 'g', label='Node 2', linewidth=2)
#plt.plot(trl, Q[:,2], 'y', label='Node 3', linewidth=2)
#plt.plot(trl, Q[:,3], 'k', label='Node 4', linewidth=2)
#plt.xlabel(r'$\lambda_{tr}$ [$^o$]', fontsize=18)
#plt.ylabel('Power [W]', fontsize=18)
#plt.title('Solar power during orbit',fontsize=30)
#plt.ylim([0,250])
#plt.xlim([0,360])
#plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
#plt.legend(loc='upper right')
##fig2 = plt.figure(2)
##plt.plot(t,trl)
##plt.xlabel('t [s]')
##plt.ylabel('trlong')
#plt.show()
###


### albedo power calculation, must uncomment 0 in solar flux function
#print ('Albedo Flux')
#t = np.linspace(0,2*orbit,2*orbit+1)
#trl = np.zeros([len(t)])
#for i in range(1,7):
    #sat[i].epsilon = 0
#Q = np.zeros([len(t),6])
#for i in range(0,6):
    #for i2 in range(0,len(t)):
        #trl[i2] = t2trlong(t[i2])
        #Q[i2,i] = Rad(sat[i+1],trl[i2])
#Q2 = SolarEff * Q /(1-SolarEff)
#E = Q.sum() * orbit / len(t) * SolarEff/(1-SolarEff)
#W = E / orbit
#print ('Total energy received = ', E, ', constant power = ', W)
#fig1 = plt.figure(1, figsize=(16,12))
#plt.plot(trl, Q[:,0]+Q[:,1]+Q[:,2]+Q[:,3]+Q[:,4]+Q[:,5], 'b', label='Total Power', linewidth=2)
##plt.plot(trl, Q[:,2], 'r', label='Total Power +Z')
##plt.plot(trl, Q2[:,0]+Q2[:,1]+Q2[:,2]+Q2[:,3]+Q2[:,4]+Q2[:,5], 'b', label='Effective Power')
#plt.plot(trl, Q[:,0], 'r', label='Nodes 1, 2, 5, 6', linewidth=2)
##plt.plot(trl, Q[:,1], 'g', label='Node 2', linewidth=2)
#plt.plot(trl, Q[:,2], 'y', label='Node 3', linewidth=2)
##plt.plot(trl, Q[:,4], 'c', label='Node 5', linewidth=2)
##plt.plot(trl, Q[:,5], 'm', label='Node 6', linewidth=2)
#plt.xlabel(r'$\lambda_{tr}$ [$^o$]', fontsize=18)
#plt.ylabel('Power [W]', fontsize=18)
#plt.title('Albedo power during orbit',fontsize=30)
#plt.ylim([0,200])
#plt.xlim([0,360])
#plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
#plt.legend(loc='upper right')
##fig2 = plt.figure(2)
##plt.plot(t,trl)
##plt.xlabel('t [s]')
##plt.ylabel('trlong')
#plt.show()
###


### IR power calculation
#print ('IR Flux')
#t = np.linspace(0,2*orbit,2*orbit+1)
#trl = np.zeros([len(t)])
#for i in range(1,7):
    #sat[i].alpha = 0
#Q = np.zeros([len(t),6])
#for i in range(0,6):
    #for i2 in range(0,len(t)):
        #trl[i2] = t2trlong(t[i2])
        #Q[i2,i] = Rad(sat[i+1],trl[i2])
#E = Q.sum() * dt
#W = E / orbit
#print ('Total energy received = ', E, ', constant power = ', W)
#fig1 = plt.figure(1, figsize=(16,12))
#plt.plot(trl, Q[:,0]+Q[:,1]+Q[:,2]+Q[:,4]+Q[:,5], 'b', label='Total Power', linewidth=2)
#plt.plot(trl, Q[:,0], 'r', label='Nodes 1,2,5,6', linewidth=2)
#plt.plot(trl, Q[:,2], 'g', label='Node 3', linewidth=2)
#plt.xlabel(r'$\lambda_{tr}$ [$^o$]', fontsize=18)
#plt.ylabel('Power [W]', fontsize=18)
#plt.title('IR power during orbit',fontsize=30)
#plt.ylim([0,40])
#plt.xlim([0,360])
#plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
#plt.legend(loc='upper right')
##fig2 = plt.figure(2)
##plt.plot(t,trl)
##plt.xlabel('t [s]')
##plt.ylabel('trlong')
#plt.show()
###


#t1 = np.arange(0,orbit+0.1,dt/100)
#data = np.zeros([len(t1)])
#for i in range(0,len(t1)):
    #trl = t2trlong(t1[i])
    #NextTempSimp(sat2,trl,Dt)
    #data[i] = sat2.T
    #sat2.T = sat2.Tnext
#fig1 = plt.figure(1, figsize=(16,12))
#plt.plot(t1, data, 'b', label='Node 1')
#plt.xlabel('t [s]', fontsize=18)
#plt.ylabel('T [K]', fontsize=18)
#plt.title('Single node',fontsize=30)
#plt.ylim([0,500])
#plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
#plt.legend(loc='upper right')
#plt.show()


#t1 = np.arange(0,5*orbit+0.1,dt)
#data = np.zeros([len(t1),2])
#for i in range(0,len(t1)):
    #trl = t2trlong(t1[i])
    #NextTempSimp2(sat,trl,dt)
    #for i2 in range(0,2):
        #data[i,i2] = sat[i2].T
        #sat[i2].T = sat[i2].Tnext
#fig1 = plt.figure(1, figsize=(16,12))
#plt.plot(t1, data[:,0], 'b', label='Node 1')
#plt.plot(t1, data[:,1], 'r', label='Node 2')
#plt.xlabel('t [s]', fontsize=18)
#plt.ylabel('T [K]', fontsize=18)
#plt.title('Two nodes',fontsize=30)
#plt.ylim([0,500])
#plt.grid(b=True, which='both', axis='both', linestyle='-', linewidth=.3)
#plt.legend(loc='upper right')
#plt.show()



t1 = np.arange(0,15*orbit+0.1,dt/5)
for i in range(0,len(t1)):
    trl = t2trlong(t1[i])
    NextTemp(sat,trl,dt/10)
    for i2 in range(0,7):
        sat[i2].T = sat[i2].Tnext

t2 = np.arange(0,orbit,dt/10)
data = np.zeros([len(t2),7])
trl = np.zeros([len(t2)])
for i in range(0,len(t2)):
    trl[i] = t2trlong(t2[i])
    NextTemp(sat,trl[i],dt/10)
    for i2 in range(0,7):
        data[i,i2] = sat[i2].T
        sat[i2].T = sat[i2].Tnext

data -= kelvin

print ('T0 = ', data[0,:])
print ('Tf = ', data[-1,:])

Plot(trl,data,-20,130,1)