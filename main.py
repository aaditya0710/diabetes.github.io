from flask import Flask,request,render_template,url_for
import pickle

app = Flask(__name__)
model = pickle.load(open("saved_model.pkl",'rb'))
print("back again")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    """Let's check for diabetes
    This is using docstring for specifications
    ---
    parameters:
        - name: Glucose
          in: query
          type: number
          required: True
        - name: Pregnancies
          in: query
          type: number
          required: True
        - name: Insulin
          in: query
          type: number
          required: True
        - name: BMI
          in: query
          type: number
          required: True
        - name: Age
          in: query
          type: number
          required: True
    responses:
        200:
            description: The result for diabetes is
    """
    glucose = int(request.form["Glucose"])
    Pregnancies = int(request.form["Pregnancies"])
    Insulin = int(request.form["Insulin"])
    Age = int(request.form["Age"])
    BMI =int(request.form["BMI"])

    prediction = model.predict([[glucose,Pregnancies,Insulin,BMI,Age]])
    string = "negative"
    if prediction==[1]:
        string = "positive"
    
    return "The result for diabetes is "+string
if __name__ == "__main__":
    app.run(debug = True)
