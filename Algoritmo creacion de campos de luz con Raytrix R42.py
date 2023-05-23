#Librerias a usar
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pandas as pd
from sklearn.cluster import KMeans
from  matplotlib.lines import Line2D

#Direcciones donde seran leidas las fotos en blanco
path1 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Fotos en blanco\RxImg_00001.jpg'
path2 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Prueba\RxImg_00002_m.jpg'
#Lectura de la foto en blanco y la foto a procesar
img_white=cv2.imread(path1)
img_Process=cv2.imread(path2)
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
"""
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

"""
ArrayCX = []
ArrayCY = []
ArrayCero = []
New_Clusters = []

img_t1 = 255*np.ones((175,150,3), np.uint8)
img_t2 = 255*np.ones((175,150,3), np.uint8)
img_t3 = 255*np.ones((175,150,3), np.uint8)

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

Clusters_Y_aux = []
n_clust = 176
Data1 = pd.DataFrame({"Ceros":ArrayCero,"EjeY":ArrayCY})
K_Means1=KMeans(n_clusters=n_clust).fit(Data1.values)
Data1["EjeX"] = ArrayCX
Data1["clusterX"] = K_Means1.labels_
for i in range (len(ArrayCY)):
    Clusters_Y_aux.append(Data1.clusterX[i])

"""
plt.figure(figsize=(6, 5), dpi=100)
colores = []
for clusterX in range(K_Means1.n_clusters):
    colores.append("green")
    colores.append("blue")
    colores.append("orange")
    colores.append("purple")
    colores.append("red")

for clusterX in range(K_Means1.n_clusters):
    plt.scatter(Data1[Data1["clusterX"] == clusterX]["Ceros"],
                Data1[Data1["clusterX"] == clusterX]["EjeY"],
                s=1, color=colores[clusterX])
    plt.xlabel("Ceros [Pixeles]")
    plt.ylabel("Coor centroides eje Y [Pixeles]")
    plt.title("Grafica de clasificación para los centroides del eje Y")
    plt.grid()
plt.savefig("Grafica centroides del eje Y.jpg", bbox_inches='tight')
plt.show()
"""

for i in range(len(Clusters_Y_aux)):
    min_id = i
    for j in range(i+1, len(Clusters_Y_aux)):
        if ArrayCY[j] < ArrayCY[min_id]:
            min_id = j
    ArrayCY[i], ArrayCY[min_id] = ArrayCY[min_id], ArrayCY[i]
    ArrayCX[i], ArrayCX[min_id] = ArrayCX[min_id], ArrayCX[i]
    Clusters_Y_aux[i], Clusters_Y_aux[min_id] = Clusters_Y_aux[min_id], Clusters_Y_aux[i]
    New_Clusters[i], New_Clusters[min_id] = New_Clusters[min_id], New_Clusters[i]
    

Clusters_Y_aux.append(-1)
tam = 0
for i in range(n_clust):
    j = 0
    while Clusters_Y_aux[tam+j] == Clusters_Y_aux[tam+j+1] and Clusters_Y_aux[tam+j+1] != -1:
        j = j+1
    j = j+1
    for a in range (j):
        min_id = a+tam
        for b in range(a+1, j):
            if ArrayCX[tam+b] < ArrayCX[min_id]:
                min_id = b+tam
        ArrayCY[tam+a], ArrayCY[min_id] = ArrayCY[min_id], ArrayCY[tam+a]
        ArrayCX[tam+a], ArrayCX[min_id] = ArrayCX[min_id], ArrayCX[tam+a]
        New_Clusters[tam+a], New_Clusters[min_id] = New_Clusters[min_id], New_Clusters[tam+a]
    tam = tam+j

Lente_tipo = ['Grande', 'Mediano', 'Pequeño']
Ord_Lentes = []
Ord_Area = []
for i in range(3):
    Ord_Area.append(Area[Clusters.index(i)])
    Ord_Lentes.append(i)
for i in range(3):
    min_id = i
    for j in range(i+1, 3):
        if Ord_Area[j] < Ord_Area[min_id]:
            min_id = j
    Ord_Area[i], Ord_Area[min_id] = Ord_Area[min_id], Ord_Area[i]
    Ord_Lentes[i], Ord_Lentes[min_id] = Ord_Lentes[min_id], Ord_Lentes[i]

for i in range(3):
    f1 = open (r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Prueba\Lightfield flores lente '+str(Ord_Lentes[i]+1)+'\Lente_tipo_'+Lente_tipo[i]+' .txt','w')
    f1.write('Lente_tipo_'+Lente_tipo[i])
    f1.close()
print(Ord_Area)
print(Ord_Lentes)

Range1 = 7#7
Range2 = 8#8
for Cont1 in range(-Range1,Range2):
    for Cont2 in range(-Range1,Range2):
        tam = 0
        otro = 0
        for i in range(n_clust-1):
            a = 0
            while Clusters_Y_aux[tam+a] == Clusters_Y_aux[tam+a+1] and Clusters_Y_aux[tam+a+1] != -1:
                a = a+1
            a = a+1
            Aux = 0
            for j in range(a):
                if New_Clusters[otro] == Ord_Lentes[0]:
                    img_t1[i][Aux][0] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][0]
                    img_t1[i][Aux][1] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][1]
                    img_t1[i][Aux][2] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][2]
                    Aux = Aux+1
                otro = otro+1
            tam = tam+a
                
        Out1 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Prueba\Lightfield flores lente 1\Img_flor_'+str(Cont1+Range1)+','+str(Cont2+Range2)+'.jpg'
        #img1Rotate = cv2.rotate(img_t1[0:175,1:73], cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(Out1,img_t1[0:175,1:73])

        tam = 0
        otro = 0
        for i in range(n_clust-1):
            a = 0
            while Clusters_Y_aux[tam+a] == Clusters_Y_aux[tam+a+1] and Clusters_Y_aux[tam+a+1] != -1:
                a = a+1
            a = a+1
            Aux = 0
            for j in range(a):
                if New_Clusters[otro] == Ord_Lentes[1]:
                    img_t2[i][Aux][0] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][0]
                    img_t2[i][Aux][1] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][1]
                    img_t2[i][Aux][2] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][2]
                    Aux = Aux+1
                otro = otro+1
            tam = tam+a

        Out2 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Prueba\Lightfield flores lente 2\Img_flor_'+str(Cont1+Range1)+','+str(Cont2+Range2)+'.jpg'    
        cv2.imwrite(Out2,img_t2[0:175,1:73])

        tam = 0
        otro = 0
        for i in range(n_clust-1):
            a = 0
            while Clusters_Y_aux[tam+a] == Clusters_Y_aux[tam+a+1] and Clusters_Y_aux[tam+a+1] != -1:
                a = a+1
            a = a+1
            Aux = 0
            for j in range(a):
                if New_Clusters[otro] == Ord_Lentes[2]:
                    img_t3[i][Aux][0] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][0]
                    img_t3[i][Aux][1] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][1]
                    img_t3[i][Aux][2] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][2]
                    Aux = Aux+1
                otro = otro+1
            tam = tam+a

        Out3 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Prueba\Lightfield flores lente 3\Img_flor_'+str(Cont1+Range1)+','+str(Cont2+Range2)+'.jpg'     
        cv2.imwrite(Out3,img_t3[0:175,1:73])



cv2.waitKey(0)
cv2.destroyAllWindows()