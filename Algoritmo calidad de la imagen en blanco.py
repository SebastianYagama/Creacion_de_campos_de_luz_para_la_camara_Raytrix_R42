#Librerias a usar
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pandas as pd
from sklearn.cluster import KMeans
from  matplotlib.lines import Line2D
from random import randint

#Direcciones donde seran leidas las fotos en blanco
path1 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Fotos en blanco\RxImg_00001.jpg'
#Lectura de la foto en blanco y la foto a procesar
img_white=cv2.imread(path1)
#Transformacion de la imagen en blanco a escala de grises
imGris=cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
#Umbralizacion de la imagen en escala de grises por rangos
frame_threshold = cv2.inRange(imGris, 80, 155)
#Deteccion de los contornos sin proximacion y con jerarquia
contours,hierarchy=cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

Area = []
Perimetro = []
contours_sons = []
for i in range (len(contours)):
    if (hierarchy[0][i][2] != -1) & (hierarchy[0][i][3] == -1):
        if(((cv2.contourArea(contours[i]))-(cv2.contourArea(contours[hierarchy[0][i][2]]))) < 400):
            if((cv2.arcLength(contours[hierarchy[0][i][2]], True)) > 40):
                Area.append((cv2.contourArea(contours[i]))-(cv2.contourArea(contours[hierarchy[0][i][2]])))
                Perimetro.append(cv2.arcLength(contours[hierarchy[0][i][2]], True))
                contours_sons.append(contours[hierarchy[0][i][2]])
Clusters = []
Data = pd.DataFrame({"Perimetro":Perimetro,"Area":Area})
K_Means=KMeans(n_clusters=3).fit(Data.values)
Data["cluster"]=K_Means.labels_
for i in range (len(contours_sons)):
    Clusters.append(Data.cluster[i])#Cambiarlo por list


legends_elements = [Line2D([0],[0],marker='o',color='w',label='Lente de tipo 1',markerfacecolor='orange',markersize=10),
                    Line2D([0],[0],marker='o',color='w',label='Lente de tipo 2',markerfacecolor='green',markersize=10),
                    Line2D([0],[0],marker='o',color='w',label='Lente de tipo 3',markerfacecolor='blue',markersize=10)]

plt.figure(figsize=(6, 5), dpi=100)
colores = ["green", "blue", "orange", "black", "purple", "pink", "brown"]
for cluster in range(K_Means.n_clusters):
    plt.scatter(Data[Data["cluster"] == cluster]["Area"],
                Data[Data["cluster"] == cluster]["Perimetro"],
                s=10, color=colores[cluster])
    plt.xlabel("Area [Pixeles^2]")
    plt.ylabel("Perimetro [Pixeles]")
    plt.title("Grafica de clasificación para los 3 lentes")
    plt.grid()
    plt.legend(handles=legends_elements, loc='upper right')
plt.savefig("Grafica con K medias.jpg", bbox_inches='tight')
plt.show()

uno=0
dos=0
tres=0

for i in range (len(Area)):
    if Data.cluster[i]==0:
        cv2.drawContours(img_white, contours_sons[i], -1,(255, 0, 0), 1)#Blue
        uno=uno+1
    if Data.cluster[i]==1:
        cv2.drawContours(img_white, contours_sons[i], -1,(0, 255, 0), 1)#Green
        dos=dos+1
    if Data.cluster[i]==2:
        cv2.drawContours(img_white, contours_sons[i], -1,(0, 0, 255), 1)#Redw
        tres=tres+1

ArrayCX = []
ArrayCY = []
ArrayCero = []
New_Clusters = []
New_Contorno = []

for i in range(len(contours_sons)):
    M=cv2.moments(contours_sons[i])
    if M["m00"] != 0:
        X=int(M["m10"]/M["m00"])
        Y=int(M["m01"]/M["m00"])
        if (Y+12) < 5364 and (X+12) < 7716:
            ArrayCX.append(X)
            ArrayCY.append(Y)
            ArrayCero.append(0)
            New_Clusters.append(Clusters[i])
            New_Contorno.append(contours_sons[i])

print(len(New_Contorno))
Tam_Mat = []
for i in range(50):
    Conteo = 0
    Salir = 0
    N_Random = (randint(0,len(New_Contorno)))
    for a in range(20):
        for j in range(len(New_Contorno[N_Random])):
            if ArrayCX[N_Random]+a == New_Contorno[N_Random][j][0][0] and ArrayCY[N_Random]+a == New_Contorno[N_Random][j][0][1]:
                Salir = 1
            if ArrayCX[N_Random]-a == New_Contorno[N_Random][j][0][0] and ArrayCY[N_Random]-a == New_Contorno[N_Random][j][0][1]:
                Salir = 1
            if ArrayCX[N_Random]+a == New_Contorno[N_Random][j][0][0] and ArrayCY[N_Random]-a == New_Contorno[N_Random][j][0][1]:
                Salir = 1
            if ArrayCX[N_Random]-a == New_Contorno[N_Random][j][0][0] and ArrayCY[N_Random]+a == New_Contorno[N_Random][j][0][1]:
                Salir = 1
        if Salir != 1:
            Conteo=Conteo+1
    Tam_Mat.append(Conteo)

N_Menor = Tam_Mat[0]
for i in Tam_Mat:
    if (N_Menor > i and i > 3):
        N_Menor = i

uno=0
dos=0
tres=0
for i in range (len(New_Clusters)):
    if New_Clusters[i]==0:
        uno=uno+1
    if New_Clusters[i]==1:
        dos=dos+1
    if New_Clusters[i]==2:
        tres=tres+1

Total_lentes = 38720
Error_total = ((abs(len(New_Contorno)-Total_lentes))/Total_lentes)*100
a =  Total_lentes/3
error1=((abs(uno-a))/a)*100
error2=((abs(dos-a))/a)*100
error3=((abs(tres-a))/a)*100

print("Se hallaron "+str(len(New_Contorno))+" lentes de "+str(38720)+" lentes. Con un error total del {:.2f}%, repartidos en:".format(Error_total))
print("Lente 1:"+ str(uno)+" con error del {:.2f}%".format(error1))
print("Lente 2:"+ str(dos)+" con error del {:.2f}%".format(error2))
print("Lente 3:"+ str(tres)+" con error del {:.2f}%".format(error3))

print("Para esta foto en blanco se recomienda usar una matriz "+str(N_Menor*2)+"x"+str(N_Menor*2)+" o menor.")






