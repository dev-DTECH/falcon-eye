import threading

from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')

def test():
    global socketio
    while True:
        print("violence")
        socketio.emit('alert-violence', {}, broadcast=True)
if __name__ == '__main__':
    # global socketio
    print("test")
    t = threading.Thread(target=test)
    t.daemon = True
    t.start()
    socketio.run(app,port=8080)

