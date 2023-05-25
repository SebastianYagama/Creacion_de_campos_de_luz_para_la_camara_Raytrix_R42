#Algoritmo para la creación de tres tipos de campos de luz usando fotografías 
#            adquiridas por la cámara plenóptica Raytrix R42.
#
#
#    Autores:
#       - Jhon Sebastian Yagama Parra
#       - Juan Sebastian Mora Zarza
#
#
#   Pontificia Universidad Javeriana
#              Bogota D.C.
#                 2023
#
#
#Descripción general del algoritmo implementado:
#
#   Este algoritmo se basa en la creación de campos de luz para los 3 tipos de micro-lentes 
#con los que cuenta la cámara Raytrix R42. Para ello, el algoritmo detecta todos 
#los micro-lentes por medio de hallar contornos, los clasifica, en el caso de la cámara Raytrix R42, 
#en tres tipos diferentes, halla todos los centros de cada lente y en base a esta información, 
#realiza la reconstrucción de las imágenes para la creación de los 3 tipos de campos de luz.
#
#   Nota 1: El algoritmo suele tardar entre 2 a 3 minutos en ejecutarse dependiendo del 
#           tamaño de la matriz para la creación del campo de luz.
#   Nota 2: Se recomienda editar solo las entradas del algortimo.
#   Nota 3: Si aparece el siguiente error: FileExistsError: ([WinError 183] No se puede 
#           crear un archivo que ya existe: 'Dirección') Se deben mover o borrar las carpetas
#           creadas para los campos de luz (Lightfield flores lente 1, Lightfield flores lente 2
#           y Lightfield flores lente 3)
#
#
#Entradas:
#   Obligatorias:
#   -   Imagen en blanco.
#   -   Imagen a generar los campos de luz.
#   -   Dirección donde serán creadas las carpetas donde serán guardados los campos de luz.
#
#   Opcionales: (Pueden mejorar los resutados, a veces son obligatorias)
#   -   Tamaño de la matriz para la creación del campo de luz.
#   -   Rangos en los que la función de thresholding o umbralización trabajará
#
#Salidas:
#   -   Tres carpetas donde se incluirán los campos de luz para los tres tipos de lentes.
#
#
#Para obtener más información leer el documento en la página de Sharepoint o en la carpeta Documentación: 
#   -   Método para la creación de un campo de luz con cámaras plenópticas.pdf
#
#
#
#Librerias a usar, algunas se deben instalar si su python los las tiene.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pandas as pd
import os
from sklearn.cluster import KMeans
from  matplotlib.lines import Line2D
#
#
#
#   Entradas Obligatorias:
#Dirección donde será leida la foto en blanco
path1 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Fotos en blanco\Foto_en_blanco_1.jpg'
#Dirección donde será leida la foto de la que se creará el campo de luz
path2 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field\Fotos a procesar\RxImg_00002.jpg'
#Dirección donde serán creadas las tres carpetas para el almacenamiento de los campos de luz
path3 = r'C:\Users\sebas\OneDrive\Escritorio\1. Tesis Plenoptica\Codigo light field'
#
#   Entradas opcionales:
#Tamaño de la matriz para la creación del campo de luz: tamaño por defecto [15x15]
Dim1 = 15
Dim2 = 15
#Rango superior e inferior que serán usados para el thresholding o umbralizacón
#Recomendacion:No usar un rango menor a 70 o mayor a 200, si se toma una foto en 
#blanco nueva se deben hallar nuevamente estos rangos
#
RangoSuperior = 150
RangoInferior = 75
#
#Sugerencias: Para 'Foto_en_blanco_1.jpg' usar 80 y 200
#Sugerencias: Para 'Foto_en_blanco_lente_59872' usar 75 y 150
#
#
#
#
#               Inicio del Algoritmo
#
#Nombres de las carpetas en las que se almacenarán los campos de luz
Carpeta1 = r'\Lightfield flores lente 1'
Carpeta2 = r'\Lightfield flores lente 2'
Carpeta3 = r'\Lightfield flores lente 3'
#Creación de las carpetas
os.mkdir(path3+Carpeta1)
os.mkdir(path3+Carpeta2)
os.mkdir(path3+Carpeta3)
#
#Lectura de la foto en blanco y la foto para la creación del campo de luz
img_white=cv2.imread(path1)
img_Process=cv2.imread(path2)
#Transformacion de la imagen en blanco a escala de grises
imGris=cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
#Umbralizacion de la imagen en escala de grises por rangos
frame_threshold = cv2.inRange(imGris, RangoInferior, RangoSuperior)
#Deteccion de los contornos sin proximacion y con jerarquia
contours,hierarchy=cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#Algoritmo para la deteccion de contornos internos o contornos hijos
Area = []
Perimetro = []
contours_sons = []
for i in range (len(contours)):
    if (hierarchy[0][i][2] != -1) & (hierarchy[0][i][3] == -1):#Condición para saber si son contornos padre o hijo
        if(((cv2.contourArea(contours[i]))-(cv2.contourArea(contours[hierarchy[0][i][2]]))) < 400):#Limite superior para el area
            if((cv2.arcLength(contours[hierarchy[0][i][2]], True)) > 40):#Limite inferior para el perimetro
                Area.append((cv2.contourArea(contours[i]))-(cv2.contourArea(contours[hierarchy[0][i][2]])))#Resta de las areas entre contorno padre e hijo
                Perimetro.append(cv2.arcLength(contours[hierarchy[0][i][2]], True))#Perimetro del contorno hijo
                contours_sons.append(contours[hierarchy[0][i][2]])#Obtención de contornos hijos

#Clasificación con K-Means:
Clusters = []
Data = pd.DataFrame({"Perimetro":Perimetro,"Area":Area})
K_Means=KMeans(n_clusters=3).fit(Data.values)
Data["cluster"]=K_Means.labels_
for i in range (len(contours_sons)):
    Clusters.append(Data.cluster[i])

#Detección de centroides para cada micro lente y almacenamiento de sus coordenadas
ArrayCX = []
ArrayCY = []
ArrayCero = []
New_Clusters = []
for i in range(len(contours_sons)):
    M=cv2.moments(contours_sons[i])#Halla el momento de cada contorno hijo
    if M["m00"] != 0:
        X=int(M["m10"]/M["m00"])#Halla la coordenada en x del centroide
        Y=int(M["m01"]/M["m00"])#Halla la coordenada en y del centroide
        if (Y+12) < 5364 and (X+12) < 7716:#Limite de centroides con foto
            ArrayCX.append(X)
            ArrayCY.append(Y)
            ArrayCero.append(0)
            New_Clusters.append(Clusters[i])

#Clasificación con K-Means para el eje X:
Clusters_Y_aux = []
n_clust = 176
Data1 = pd.DataFrame({"Ceros":ArrayCero,"EjeY":ArrayCY})
K_Means1=KMeans(n_clusters=n_clust).fit(Data1.values)
Data1["EjeX"] = ArrayCX
Data1["clusterX"] = K_Means1.labels_
for i in range (len(ArrayCY)):
    Clusters_Y_aux.append(Data1.clusterX[i])

#Reordenamiento para el eje Y
for i in range(len(Clusters_Y_aux)):
    min_id = i
    for j in range(i+1, len(Clusters_Y_aux)):
        if ArrayCY[j] < ArrayCY[min_id]:
            min_id = j
    ArrayCY[i], ArrayCY[min_id] = ArrayCY[min_id], ArrayCY[i]
    ArrayCX[i], ArrayCX[min_id] = ArrayCX[min_id], ArrayCX[i]
    Clusters_Y_aux[i], Clusters_Y_aux[min_id] = Clusters_Y_aux[min_id], Clusters_Y_aux[i]
    New_Clusters[i], New_Clusters[min_id] = New_Clusters[min_id], New_Clusters[i]
    
#Reordenamiento para el eje X
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

#Clasificación de lente en un .txt para cada carpeta
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
    f1 = open (path3+r'\Lightfield flores lente '+str(Ord_Lentes[i]+1)+'\Lente_tipo_'+Lente_tipo[i]+' .txt','w')
    f1.write('Lente_tipo_'+Lente_tipo[i])
    f1.close()

#Creación de imágenes en blanco para sobreescribir en la reconstrucción
img_t1 = 255*np.ones((175,75,3), np.uint8)
img_t2 = 255*np.ones((175,75,3), np.uint8)
img_t3 = 255*np.ones((175,75,3), np.uint8)

#Ajuste de rangos de la matriz para la creación de los campos de luz
Dim1_Range1 = int(Dim1/2)
if Dim1 % 2 == 0:
    Dim1_Range2 = Dim1_Range1
else:
    Dim1_Range2 = Dim1_Range1+1
Dim2_Range1 = int(Dim2/2)
if Dim2 % 2 == 0:
    Dim2_Range2 = Dim2_Range1
else:
    Dim2_Range2 = Dim2_Range1+1

#Reconstrucción y creación de los campos de luz
for Cont1 in range(-Dim1_Range1,Dim1_Range2):
    for Cont2 in range(-Dim2_Range1,Dim2_Range2):
        tam = 0
        otro = 0
        for i in range(n_clust-1):
            a = 0
            while Clusters_Y_aux[tam+a] == Clusters_Y_aux[tam+a+1] and Clusters_Y_aux[tam+a+1] != -1:
                a = a+1
            a = a+1
            Aux = 0
            for j in range(a):
                if New_Clusters[otro] == Ord_Lentes[0] and Aux < 75:
                    img_t1[i][Aux][0] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][0]
                    img_t1[i][Aux][1] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][1]
                    img_t1[i][Aux][2] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][2]
                    Aux = Aux+1
                otro = otro+1
            tam = tam+a
                
        Out1 = (path3+r'\Lightfield flores lente 1\Img_flor_'+str(Cont1+Dim1_Range2)+','+str(Cont2+Dim2_Range2)+'.jpg')
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
                if New_Clusters[otro] == Ord_Lentes[1] and Aux < 75:
                    img_t2[i][Aux][0] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][0]
                    img_t2[i][Aux][1] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][1]
                    img_t2[i][Aux][2] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][2]
                    Aux = Aux+1
                otro = otro+1
            tam = tam+a

        Out2 = path3+r'\Lightfield flores lente 2\Img_flor_'+str(Cont1+Dim1_Range2)+','+str(Cont2+Dim2_Range2)+'.jpg'    
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
                if New_Clusters[otro] == Ord_Lentes[2] and Aux < 75:
                    img_t3[i][Aux][0] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][0]
                    img_t3[i][Aux][1] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][1]
                    img_t3[i][Aux][2] = img_Process[ArrayCY[otro]+Cont1][ArrayCX[otro]+Cont2][2]
                    Aux = Aux+1
                otro = otro+1
            tam = tam+a

        Out3 = path3+r'\Lightfield flores lente 3\Img_flor_'+str(Cont1+Dim1_Range2)+','+str(Cont2+Dim2_Range2)+'.jpg'     
        cv2.imwrite(Out3,img_t3[0:175,1:73])

cv2.waitKey(0)
cv2.destroyAllWindows()