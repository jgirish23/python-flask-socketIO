# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, url_for, request, jsonify, redirect
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/test')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'

check = 1
cnt = 1
@socketio.on('message')
def handle_message(message):
    print('Received message:', message)
    # if(cnt%2 == 0):
    global curr_input, cnt
    curr_input = message["someData"]
	# Handle the message and send a response if needed
	# response = {'status': 'OK'}
    # socketio.emit('server_message', {'message': "output"})
    print(f"out: {cnt}")
    out = input()
    socketio.emit('server_message', {'message': out})
    cnt = cnt +1
    out = input()
    socketio.emit('server_message', {'message': out})
    # socketio.emit('response', {"status": "OK"})
    global check



@app.route('/',methods = ['GET','POST'])
def home():
	if( request.method == 'POST' ):
		print(request.json["user_message"])
		print("post req!!!!!")
		response = jsonify({
            "stuff": "Here be stuff"
        })

		return response, 200

	return render_template("indexx_1.html")



# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.

	# getInfo()
	print("----------------------------------------------------------------------------------------")
	socketio.run(app)

	# app.run(debug=True)
