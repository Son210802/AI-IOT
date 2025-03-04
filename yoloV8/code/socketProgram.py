import socket
import yaml
import time

class SocketConnection:
    def __init__(self, host, port=3000):
        ''' Socket connection with 2 param
            host: your ip address host pc
            port: initiate port no above 1024. Default is 3000
        '''
        self.host = host
        self.port = port
        self.instance = socket.socket()  # instantiate
        self.instance.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def clientConnect(self):
        ''' Parse connect client socket via host and port
        '''
        try:
            self.instance.connect((self.host, self.port))
            print(f"Connected to {self.host} on port {self.port}")
        except socket.error as e:
            print(f"Failed to connect to {self.host} on port {self.port}: {e}")
            time.sleep(5)
            self.clientConnect()

    def clientSend(self, message):
        # message = input("send: ")  # take input
        self.instance.send(message.encode())  # send message
    
    def clienRcv(self):
        return self.instance.recv(1024).decode()  # receive response

    def serverBind(self, num=2):
        # bind host address and port together
        # The bind() function takes tuple as argument
        # default configure number client the server can listen simultaneously
        # global conn
        try:
            self.instance.bind((self.host, self.port))
            # configure how many client the server can listen simultaneously
            self.instance.listen(num)
            self.conn, address = self.instance.accept()  # accept new connection
            print("Connection from: " + str(address))
        except socket.error as e:
            print(f"Failed to connect to {self.host} on port {self.port}: {e}")
            time.sleep(5)
            self.serverBind()
    
    def serverRcv(self):
        return self.conn.recv(1024).decode() # Receive response from client
            
    def serverSend(self, message):
        # data = input('send: ')
        self.conn.send(message.encode())  # send message to the client

    def send(self, connType, connector, message):
        if connType == "server":
            connector.serverSend(message)  # send message to the client
        elif connType == "client":
            connector.clientSend(message)   # send message to the server

    def close(self):
        self.instance.close()
        print("\nSocket closed!")

if __name__ == '__main__': 
    # Them socket vao model
    host = "192.168.1.134" #IP address from your host pc 192.168.1.2
    port = 3000  # socket server port number

    with open('config.yml', 'r') as f:
        items = yaml.load(f, Loader=yaml.FullLoader)

    # Test connect client with server role
    server = SocketConnection(items['Host'], 3000)   # 192.168.1.10
    server.serverBind()
    try:
        data = server.serverRcv()
        print("Reveived from client: " + str(data))
        mess = "ba"
        server.send("server", server, str(mess))
    finally:
        server.close()
