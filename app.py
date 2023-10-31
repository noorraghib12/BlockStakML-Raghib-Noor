from flask import (Flask, 
                   request,
                   current_app,
                   jsonify,
                   render_template, 
                   request, 
                   flash,
                   redirect, 
                   url_for,
                   current_app)

from joblib import load
import pandas as pd



app=Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('customer_profile'))

@app.route('/customer_profile',methods=["GET","POST"])
def customer_profile():
    if request.method=='POST':
        entry=pd.DataFrmae(request.form)
        #INCOMPLETE
        #send entry to preprocessor from utils
        #use best model to predict, send prediction
        #return jsonify(results)
    return render_template('base.html')






if __name__=="__main__":
    app.run(debug=True,load_dotenv=True,host='localhost',port=4000)
    