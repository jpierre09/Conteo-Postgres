import cv2
import csv
import collections
import numpy as np
from tracker import *
from datetime import date
from datetime import datetime, time, timedelta
import os
import psycopg2


########## By: Pierre ##################
'''
Bitacora de mejoras y cambios:
- 23 marzo, implementación streaming con RTSP
- 24 marzo, git del codigo https://github.com/jpierre09/Vehiculos
- 28 marzo, tamaño adecuado para las lineas de deteccion 
- 3 abril, se define estructura y formato del archivo
- 4 abril, almacenamiento dinamico por hora
- 5 abril, campo fecha en el csv
- 6 abril, carpeteo automatico para almacenar data
- 15 abril, nueva ruta de almacenamiento /var/data1/ConteoVehicular/
- 15 abril, cron ejecución horaria
'''
########################################
print('[INFO] Inicializando el programa...')


#Fecha-horas
fecha_actual = datetime.now()

# Fecha, hora y minuto
fh = fecha_actual.strftime('%y/%m/%d %H:%M')

# Fecha
fhdata = fecha_actual.strftime('%y-%m-%d')

 # Hora-minuto
fechafile = fecha_actual.strftime('%H:%M')

# Minuto
fechamin = str(fecha_actual.strftime('%M'))

print('[INFO] Fecha/Hora de ejecución ' + fh)


# Ruta de almacenamiento general
#stordir = '/home/pierre/siata/codes/data/VelodromoMedellin/'
stordir = '/home/pierre/codes/ConteoVehicular/data/'
readvideo = '/home/pierre/codes/ConteoVehicular/videos/'
hora = fecha_actual.strftime('%H')
video = '/puentegirardota.mkv'

print('Fecha y hora actual: ' + readvideo+fhdata+'-'+hora+video)


minus1 = '%02d' % int('1')
#print(minus1)

horamenos = int(hora) - int(minus1)
print(horamenos)

range = range(0, 9, 1)

if horamenos == int(-1):
    diamenos = fecha_actual.strftime('%y-%m-%d')
    y = diamenos[-2:]
    z = int(y) - 1
    fhdata = diamenos[:-2] + str(z)
    print(fhdata)
    #resta = h
    #fhdata == h
    #print('Fecha restada: ' + fhdata)

    resta = '23'
    #print(resta)
    
elif horamenos in range:
    resta = "%02d" % (horamenos,)
    print(resta)
    print('Entró a este elif')
    
else:
    resta = horamenos
    print('Entró a ninguno, se conserva la resta sin tratamiento')



print('Fecha y hora menos 1 : ' + readvideo + fhdata + '-' + str(resta) + video)

#

datadir = stordir + fhdata
print('Nombre de la carpeta para crear'+datadir)

# Crea el directorio segun la fecha (Carpeteo)
try: 
    if not os.path.exists(datadir):
        os.mkdir(datadir)
        print(datadir)
except:
    print('Error al crear carpeta')
    pass


# Invoca el tracker
tracker = EuclideanDistTracker()

# Video o streaming
#cap = cv2.VideoCapture('http://10.32.38.120//axis-cgi/mjpg/video.cgi')
#cap = cv2.VideoCapture('rtsp://10.32.38.99/axis-media/media.amp')
#cap = cv2.VideoCapture('rtsp://50.32.32.2/axis-media/media.amp')
#cap = cv2.VideoCapture('rtsp://root:CamPass.21@10.32.38.99/axis-media/media.amp')
#cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
cap = cv2.VideoCapture(readvideo + fhdata + '-' + str(resta) + video)
#cap = cv2.VideoCapture('video4.avi')
input_size = 320

# Nivel minimo de confianza para el conteo
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Lineas medias de deteccion
middle_line_position = 225
up_line_angle = 200
middle_line_angle = 210
down_line_angle = 220
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15

# DB coco
classesFile = "/home/pierre/codes/ConteoVehicular/coco.names"
classNames = open(classesFile).read().strip().split('\n')
#print(classNames)
#print(len(classNames))

# Clases a detectar
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Modelo
modelConfiguration = '/home/pierre/codes/ConteoVehicular/yolov3-320.cfg'
modelWeigheights = '/home/pierre/codes/ConteoVehicular/yolov3-320.weights'

# Red convulcional
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Activar GPU

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)


# Colores random para las clases
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Encuentra el centro del rectangulo
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# Pa al almacenar las clases detectadas
temp_up_list = []
temp_down_list = []
up_list =[0, 0, 0, 0]
down_list = [0, 0, 0, 0]

# CONTEO
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Encuentra el centro del rectangulo
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Posicion actual del vehiculo
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Dibuja un mini circulo en el centro del rectangulo
    cv2.circle(img, center, 2, (0, 0, 255), -1)  
    #print(up_list, down_list)


# Funcion que da salida a los objetos detectados
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    #  Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    #print(indices)
    #print(type(indices))
    if (isinstance(indices, np.ndarray)) and (len(indices) > 0):
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        	#print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Dibuja nombre de clase y puntaje de confianza 
            #cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
            #          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Este es el limite del rectangulo
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Actualiza el seguimiento de cada clase
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime():
    #sw_test = 0
    #while sw_test < 10:
    #    sw_test = sw_test + 1
    #    fecha1 = datetime.utcnow()
    while True:
        fecha1 = datetime.utcnow()
        sucess, img = cap.read()
        if img is None:
            print(up_list)
            print('Proceso terminado')
            break
        img = cv2.resize(img,(0,0),None,0.5,0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        

        # Entrada de la red
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        
        # Alimenta la data de la red
        outputs = net.forward(outputNames)
    
        ## Encuentra los objetos de la salida de la red
        postProcess(outputs,img)

        # Las lineas del video (Corregir esas hptas)
        cv2.line(img, (0, middle_line_position), (iw, middle_line_angle), (255, 0, 255), 1)
        cv2.line(img, (0, up_line_position), (iw, up_line_angle), (0, 0, 255), 1)
        cv2.line(img, (0, down_line_position), (iw, down_line_angle), (0, 0, 255), 1)

        
        #### Panel en video, muestra el resumen de los datos
        cv2.putText(img, "Out", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "In", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Carro:     "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Moto:      "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:       "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:     "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)



        # Imprime la pantalla para inciar el conteo
        cv2.imshow('Conteo Vehicular Girardota', img)

 
        fecha2 = datetime.utcnow()
        print((fecha2 - fecha1).total_seconds()) 
        if cv2.waitKey(1) == ord('q'):
            break
        
        dir_out = ['out']
        dir_in = ['in']
        #print(pr + up_list)
        #print(pr1 + down_list) 

        #try:            
        #    with open(datadir + '/' + 'cvgir_'+ str(resta) + '.csv', 'w') as data:
        #        cwriter = csv.writer(data)
        #        #dd = [now.strftime('%y/%m/%d %H:%M')]
        #        cwriter.writerow(['Direction','car', 'bike', 'bus', 'truck'])
        #        #up_list.insert(0, 'out')
        #        #down_list.insert(0, 'in')
        #        dir1 = dir_out + up_list 
        #        dir2 = dir_in + down_list
        #        cwriter.writerow(dir1)
        #        cwriter.writerow(dir2)
        #    data.close()
        #except:
#       #     print('hola')
        #    break

        ####### PRUEBAS PARA ALMACENAMIENTO EN DB
        try:
            connection = psycopg2.connect(
                host='localhost',
                user='postgres',
                password='siata',
                database='siata_vi'
            )  

            #with open(datadir + '/' + 'cvgir_'+ str(resta) + '.csv', 'w') as data:
            #    cwriter = csv.writer(data)
                #dd = [now.strftime('%y/%m/%d %H:%M')]
            #    cwriter.writerow(['Direction','car', 'bike', 'bus', 'truck'])
                #up_list.insert(0, 'out')
                #down_list.insert(0, 'in')
            #    dir1 = dir_out + up_list 
            #    dir2 = dir_in + down_list
            #    cwriter.writerow(dir1)
            #    cwriter.writerow(dir2)
            #data.close()
            cursor=connection.cursor()
            sql= 'INSERT INTO conteo_vehicular values ( %s, %s, %s, %s)'
            dy = 'in'+up_list
            cursor.execute(sql, dy)
            connection.commit()
            connection.close
            #cursor.execute("SELECT into conteo_vehicular values('out',34,45,56,76,'')")
            #cursor.execute("SELECT version()")
            #row=cursor.fetchone()
            #print(row)
            #cursor.execute('SELECT * FROM conteo_vehicular')
            #rows=cursor.fetchall()
            #for row in rows:
            #    print(row)
            print('Conexion exitosa a DB')
        except Exception as ex:
            print('falló el almacenamiento')
        finally:
            connection.close()
            print('conexión a DB finalizada')
            #break

        ###### END DB STORAGE    

        

        # Aqui lanza la captura del objeto y destruye todas las ventanas al oprimir 'q'
    
    cap.release()
    cv2.destroyAllWindows()

####### PROCESO PARA IMAGEN ESTATICA

# image_file = '1.jpeg'
# def from_static_image(image):
#     img = cv2.imread(image)

#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

#     # Prepara la salida de la red
#     net.setInput(blob)
#     layersNames = net.getLayerNames()
#     outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
#     # Alimenta la red
#     outputs = net.forward(outputNames)

#     # Detecta los objetos del output
#     postProcess(outputs,img)

#     # cuenta la frecuencia
#     frequency = collections.Counter(detected_classNames)
#     print(frequency)
#     # Draw counting texts in the frame
#     cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#     cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#     cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#     cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)


#     cv2.imshow("image", img)

#     cv2.waitKey(0)

#     # save 
#     with open("static-data.csv", 'a') as f1:
#         cwriter = csv.writer(f1)
#         cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
#     f1.close()


    
if __name__ == '__main__':
    realTime()