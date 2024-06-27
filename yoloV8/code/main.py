import cv2

from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from socketProgram import SocketConnection
from YoloUtil import YOLOInference

import yaml
import sys
    
import logging

def threadPredict(path, connType, connector):
    # Start unferencing
    results = modelDeploy.predict(path)
    if results is not None:
        cls, angle = results
        connector.send(connType, connector, f"C_{cls} A_{angle}")
        modelDeploy.display(cls, angle)
    else:
        print("Can't infer picture!\n")
        print("Try again!\n")
        # threadCap(connType, connector)
        executor.submit(threadCap, connType, connector)
    # print(f'Done. ({time.time() - t0:.3f}s)')

def threadCap(connType, connector):
    print("\nSensor detected!")
    cap = cv2.VideoCapture(4) # Number cam is 4
    access, img = cap.read()

    # Release buffer
    cap.release()

    if access:
        # cv2.imwrite("./image/sample.jpg", img)
        # Give the thread inference
        Thread(target=threadPredict, args=(img, connType, connector)).start()
        # Give the thread inference
        # executor.submit(threadPredict, img, connType, connector).result()
    else:
        print("Failed to capture image.")
        threadCap(connType, connector)
    
def configYmlFile():
    '''
        - This is 2 method read file yaml config from local disk
        - Number 1 is given argrument via terminal console
        - Number 2 is loaded from local disk via function
    '''
    if len(sys.argv) <= 1:
        with open('config.yml', 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        stream = open(sys.argv[1], 'r')
    return yaml.load(stream, Loader=yaml.FullLoader)

def serverProgram():
    print("Current role is server!")
    # Make connection
    server = SocketConnection(items['Host'], int(items['Port']))

    server.serverBind()
    while True:
        try:                 
            data = server.serverRcv()
            if data.strip().lower() == items['Receive']:
                # Wait for the task to complete
                executor.submit(threadCap, "server", server)
            else:
                # server.serverSend(items['Send'])
                print("Incorrect message!\n")
                server.close() 
                serverProgram()
        except Exception as e:
            # server.serverSend(items['Send'])
            print(e)
        except KeyboardInterrupt:
            server.close() # Close connection
            break
    
    # server.close() # Close connection

def clientProgram():
    print("Current role is client!")
    # Make connection
    client = SocketConnection(items['Host'], int(items['Port']))
    client.clientConnect()
    while True:
        try:             
            data = client.clienRcv() # Listening signal from server
            if data.strip().lower() == items['Receive']:                
                # Thread(target=threadCap, args=("server", server)).start()
                # Wait for the task to complete
                executor.submit(threadCap, "client", client)
            else:
                #client.clientSend(items['Send'])
                print("Incorrect message!\n")
                clientProgram()
        except Exception as e:
            # client.clientSend(items['Send'])
            print(e)
        except KeyboardInterrupt:
            client.close() # Close connection
            break
    
    # client.close() # Closed connection

def main():   
    print("\nStarting demo now! Press CTRL+C to exit\n")
    if items['Name'] == "server":
        serverProgram()        
    else:
        clientProgram()
    print("\nEnd.")


if __name__ == '__main__':
    # Set logging level to ERROR for the Ultralytics module
    logging.getLogger('cv2').setLevel(logging.ERROR)

    # Dinh nghia duong dan label
    labelPath = "./label/label.txt" 

    # Path of pretrained model
    preTrainedModelPath = "./model/best.engine"

    # Initialize the thread pool executor
    executor = ThreadPoolExecutor(max_workers=4)

    # khoi tao doi tuong
    modelDeploy = YOLOInference(preTrainedModelPath, labelPath)

    # [{Name: 'server'}, {Receive: 'a'}, {Send: 'b'}, {Port: 3000}, {Host: '192.168.137.226'}]
    items = configYmlFile()
    main()