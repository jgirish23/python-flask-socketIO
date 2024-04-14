# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, url_for, request, jsonify, redirect
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
import time
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")



import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print (scores.mean())


model=SVC()
model.fit(x_train,y_train)
print("for svm: ")
print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)




def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


# def getInfo():
#     print("-----------------------------------HealthCare ChatBot-----------------------------------")
#     print("\nYour Name? \t\t\t\t",end="->")
#     name=input("")
#     print("Hello, ",name)

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

cnt = 0
curr_input = ""
curr_output = "Your Name? \t\t\t\t"

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []
    global curr_output, curr_input
    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        curr_output = "Enter the symptom you are experiencing  \t\t"
        socketio.emit('server_message', {'message': curr_output})
        # disease_input = input("")
        disease_input = ""
        print(f"Curr_input:  {curr_input}")
        while(curr_input == None):
            disease_input = curr_input
        print("Curr_input: " + disease_input)
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            socketio.emit('server_message', {'message': "searches related to input: "})
            for num,it in enumerate(cnf_dis):
                print(num,")",it)

                curr_output = str(num) + ")" + str(it)
                socketio.emit('server_message', {'message': curr_output})
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                # global curr_output
                curr_output = "Select the one you meant (0 - " + str(num) + "):  "
                socketio.emit('server_message', {'message': curr_output})
                # conf_inp = int(input(""))
                conf_inp = 0

                curr_input = None
                while(curr_input == None):
                    try:
                        # while type(curr_input) != int:
                        conf_inp = int(curr_input)
                    except Exception as e:
                        print(e)
                        # if(curr_input == None):
                        #     curr_input = None
                        # socketio.emit('server_message', {'message': "Enter valid input."})

                print("Curr_input: " ,conf_inp)
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            socketio.emit('server_message', {'message': "Enter valid symptom."})
            curr_input = None
            print("Enter valid symptom.")

    socketio.emit('server_message', {'message': "Okay. From how many days ? : "})
    curr_input = None
    while True:
        try:
            # num_days=int(input("Okay. From how many days ? : "))
            num_days = 0
            while(curr_input == None):
                # try:
                num_days = int(curr_input)
                # except Exception as e:
                #     print(e)
                #     curr_input = None
                    # socketio.emit('server_message', {'message': "Enter valid input."})
            # num_days = int(curr_input)
            break
        except:
            print("Enter valid input.")
            # curr_input = None
            # socketio.emit('server_message', {'message': "Enter valid input."})
    def recurse(node, depth):
        global curr_output, curr_input
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            socketio.emit('server_message', {'message': "Are you experiencing any "})
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""

                print(syms,"? : ",end='')
                # global curr_output
                curr_output = str(syms) + "? : "
                socketio.emit('server_message', {'message': curr_output})
                while True:
                    # inp=input("")
                    curr_input = None
                    inp = ""
                    while(curr_input == None):
                        inp = curr_input
                    # inp = curr_input
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                        curr_input = None
                        socketio.emit('server_message', {'message': "provide proper answers i.e. (yes/no) : "})
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # global curr_output
                curr_output = "You may have " + str(present_disease[0])
                socketio.emit('server_message', {'message': curr_output})

                curr_output = "" + str(description_list[present_disease[0]])
                socketio.emit('server_message', {'message': curr_output})
                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                # global curr_output
                curr_output = "You may have " + str(present_disease[0]) + "or " + str(second_prediction[0])
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                socketio.emit('server_message', {'message': curr_output})

                curr_output = str(description_list[present_disease[0]])
                print(description_list[present_disease[0]])
                socketio.emit('server_message', {'message': curr_output})

                curr_output = str(description_list[second_prediction[0]])
                print(description_list[second_prediction[0]])
                socketio.emit('server_message', {'message': curr_output})

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            socketio.emit('server_message', {'message': "Take following measures : "})
            for  i,j in enumerate(precution_list):
                # global curr_output
                print(i+1,")",j)
                curr_output = str(i+1) + ")" + str(j)
                socketio.emit('server_message', {'message': curr_output})

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)




# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/test')
def hello_world():
	return 'Hello World'

check = 1
@socketio.on('message')
def handle_message(message):
    print('Received message:', message)
    # if(cnt%2 == 0):
    global curr_input
    curr_input = message["someData"]
	# Handle the message and send a response if needed
	# response = {'status': 'OK'}
    socketio.emit('server_message', {'message': curr_output})
    print('Server message:', curr_output)
    # cnt = cnt +1
    # socketio.emit('response', {"status": "OK"})
    global check
    if check:
        check = 0
        curr_input = None
        tree_to_code(clf,cols)



@app.route('/',methods = ['GET','POST'])
def home():
	if( request.method == 'POST' ):
		print(request.json["user_message"])
		# print("post req!!!!!")
		user = jsonify({"data": 478395})
        # Set CORS headers
		response = jsonify({
            "stuff": "Here be stuff"
        })
		response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust this as needed
		response.headers.add('Access-Control-Allow-Methods', 'POST')
		response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
		return response, 200

	return render_template("index.html")

getSeverityDict()
getDescription()
getprecautionDict()

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.

	# getInfo()
	print("----------------------------------------------------------------------------------------")
	socketio.run(app)

	# app.run(debug=True)
