from flask import Flask,request
app=Flask(__name__)

@app.route('/',methods=['GET'])
def Print():
    return "Hello1 world",220
@app.route('/',methods=['POST'])
def print1():
    return "Hello",200
if  __name__=="__main__":
    app.run(port=8000,use_reloader=True)
