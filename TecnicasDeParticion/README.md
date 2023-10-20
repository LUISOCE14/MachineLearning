El codigo contine diferentes funciones con diferentes tecnicas de particion de data set que es una parte importante el los algoritmos de machine learning.

En este proyecto me toco desarrollar las diferentes tecnicas de particion de data set como: entrenamiento,prueba y validacion(en algunos casos).

Primero importe las librerias que voy usar y despues inicialize los parametros y carge  el data set,podemos partirlo para poder facilitarno el trabajo.


Cree 5 metodos de particion de datos 

#Metodo 1:
Es muy sencillo,para este metodo use las misma funciones de los data frame de pandas
    La funcion recibe 3 parametros el porcentaje de entrenamiento,el arreglo X y el arreglo y.
    Calcula el numero de filas que contendra el sub-dataset de entrenamiento y tambien calcula 
    el de el sub-dataset de entrenamiento.
    Asigna las filas y columnas correspodientes al array que le corresponde 
      Utilize la tecnica de rebanado para los dataframe, como ya tenia el numero de filas que         debia tomar el data set de entrenamiento solo se coloco el numero y automaticamente pasa 
      los datos corrrespodientes.
    Y asi con cada array que voy a retornar (X_train, X_test, y_train, y_test)


#Metodo 2
El metodo 2 es un poco mas dificil pero igual utilizando la funciones de pandas y este metodo baraja el data set para una mejor distribucion.
   La funcion recibe 3 parametros el porcentaje de entrenamiento,el arreglo X y el arreglo y.
   Primero concatena los dos arreglos que llegaron a la funcion, una vez que estan concatenados 
   utiliza la funcion sample para barajar el data set.
   Una vez barajado vuelve a partir el data sat barajado en 2 caracteristicas(X) y target(y). 
   Despues Calcula el numero de filas que contendra el sub-dataset de entrenamiento y tambien 
   calcula el  de el sub-dataset de entrenamiento.
   Por ultimo Asigna las filas y columnas correspodientes al array que le corresponde 
      Utilize la tecnica de rebanado para los dataframe, como ya tenia el numero de filas que         debia tomar el data set de entrenamiento solo se coloco el numero y automaticamente pasa 
      los datos corrrespodientes.
    Y asi con cada array que voy a retornar (X_train, X_test, y_train, y_test)


#Metodo 3
Este cambia un poco al anterior lo unico que tenemos de nueve es que este no utiliza la tecnica de rebanado si no la hace manual.

   La funcion recibe 3 parametros el porcentaje de entrenamiento,el arreglo X y el arreglo y.
   Primero concatena los dos arreglos que llegaron a la funcion, una vez que estan concatenados 
   utiliza la funcion sample para barajar el data set. 
   Despues Calcula el numero de filas que contendra el sub-dataset de entrenamiento y tambien  
   calcula el  de el sub-dataset de entrenamiento.
   Ahora crea diferentes arrays para guardar las informacion segun la condicion.
   Entra a un ciclo for  que recorrera todo el data set original.
   Dentro del data set hay una condicion y es que si la variable i es menor que el numero de       filas que le corresponden al dataset de entrenamiento.
       Mientras sea menor toda la informacion se mete en los arrays de entrenamiento
       Y si es mayor todo se ira a los data set de prueba.
   Ya que sale del ciclo convierte todo a  dataframe de pandas y lo retorna


#Metodo 4
Este metodo de particion cambia en todo puesto que este puede dividir el data set en 3 que es entrenamiento,prueba y validacion y aqui se vuelve a utilizar la tecnica de rebanado.
   La funcion recibe 4 parametros el porcentaje de entrenamiento,el arreglo X y el arreglo y y     en cuantas partes se debe particionar.
   Primero concatena los dos arreglos que llegaron a la funcion, una vez que estan concatenados 
   utiliza la funcion sample para barajar el data set.
   Despues Calcula el numero de filas que contendra el sub-dataset de entrenamiento y tambien  
   calcula el  de el sub-dataset de entrenamiento.
   Una vez calcula el numero de filas, vuelve a partir el data sat barajado en 2 
   caracteristicas(X) y target(y).
   Ahora hay una condicional,el parametro de particion es dos
     Hace el particionamiento en 2 entrenamiento y prueba.
   En cambio si el parametro es 3
     Aqui las filas que se le asignaran a el de validacion son las sobras
     Hace el particionamiento en 3 entrenamiento,prueba y validacion.
   Esta funcion retorna los array resultantes en cada condicional



Metodo 5
Este metodo es conocido, es el kfolds es muy usado en el machine learning.
     La funcion recibe 2 parametros el data set completo y las particiones.
     Primero calcula el numero de filas por cada fold
     Despues se declara una lista de folds
     Entra al ciclo for donde intera segun el numero de folds que se designo en los prametros.
        calcula para determinar los índices que se usarán como datos de prueba en esta         
        iteración.
        Se extraen de datos usando los índices de prueba_inicio a prueba_fin.
        Datos_entrenamiento se forma concatenando las partes de datos antes y después del fold de prueba. 
        Estos datos se utilizan como datos de entrenamiento en esta iteración específica.
        En cada iteracio guada el data set de entrenamiento y prueba en un fold
    Una vez termina develve la lista con todos los folds

Funcion Graficar
     La funcion recibe como parametros los diferentes arreglos creador apartir de algun metodo descrito con anterioridad
     y grafica con puntos como se ven los diferentes data set de entrenamiento,prueba y validacion.Lo hace en forma de comparacion.

  
