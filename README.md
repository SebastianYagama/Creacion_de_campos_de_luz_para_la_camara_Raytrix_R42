# Creacion_de_campos_de_luz_para_la_camara_Raytrix_R42
Este código desarrollado en Python se diseñó para la clasificación de lentes en plenópticas 2.0 y la creación de campos de luz con fotografías tomadas por la cámara Raytrix R42

Algoritmo para la creación de tres tipos de campos de luz usando fotografías 
            adquiridas por la cámara plenóptica Raytrix R42.


    Autores:
       - Jhon Sebastian Yagama Parra
       - Juan Sebastian Mora Zarza


   Pontificia Universidad Javeriana
              Bogota D.C.
                 2023


Descripción general del algoritmo implementado:

   Este algoritmo se basa en la creación de campos de luz para los 3 tipos de micro-lentes con los que cuenta la cámara Raytrix R42. Para ello, el algoritmo detecta todos los micro-lentes por medio de hallar contornos, los clasifica, en el caso de la cámara Raytrix R42, en tres tipos diferentes, halla todos los centros de cada lente y en base a esta información, realiza la reconstrucción de las imágenes para la creación de los 3 tipos de campos de luz.

   Nota 1: El algoritmo suele tardar entre 2 a 3 minutos en ejecutarse dependiendo del tamaño de la matriz para la creación del campo de luz.
   Nota 2: Se recomienda editar solo las entradas del algortimo.
   Nota 3: Si aparece el siguiente error: FileExistsError: ([WinError 183] No se puede crear un archivo que ya existe: 'Dirección') Se deben mover o borrar las carpetas creadas para los campos de luz (Lightfield flores lente 1, Lightfield flores lente 2 y Lightfield flores lente 3)


Entradas:
   Obligatorias:
   -   Imagen en blanco.
   -   Imagen a generar los campos de luz.
   -   Dirección donde serán creadas las carpetas donde serán guardados los campos de luz.

   Opcionales: (Pueden mejorar los resutados, a veces son obligatorias)
   -   Tamaño de la matriz para la creación del campo de luz.
   -   Rangos en los que la función de thresholding o umbralización trabajará

Salidas:
   -   Tres carpetas donde se incluirán los campos de luz para los tres tipos de lentes.


Para obtener más información leer el documento en la página de Sharepoint o en la carpeta Documentación: 
   -   Método para la creación de un campo de luz con cámaras plenópticas.pdf
