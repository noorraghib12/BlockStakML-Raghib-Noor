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
from utils import YesNoBinarize
import joblib
import pandas as pd


preprocessor=joblib.load('preprocessor.joblib')
tree=joblib.load('tree.joblib')
result_dict={1:"yes",0:"no"}


app=Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('customer_profile'))

@app.route('/customer_profile',methods=["GET","POST"])
def customer_profile():
    if request.method=='POST':
        entry=pd.DataFrame(request.form,index=[0])
        entry_processed=preprocessor.transform(entry)
        pred=tree.predict(entry_processed)[0]
        return {'result':result_dict[pred]}     
    return render_template('base.html')






if __name__=="__main__":
    app.run(debug=True,load_dotenv=True,host='localhost',port=4000)
    