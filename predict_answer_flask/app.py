import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
#import torchvision
from model import function_main
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    id=""
    paragraph=""
    question=""
    model_name=""
    # int_features = [str(x) for x in request.form.values()]
    #print(len(request.form.values()))
    id="10000"
    # paragraph=int_features[0]
    # question=int_features[1]
    paragraph=request.form["Context_Information"]
    question=request.form["Your_Question"]
    answer=""
    probs=0.0
    
   # model_name=int_features[2]
    answer, probs=function_main(id, paragraph, question)
    
   # s=""
   # s=answer + '\n' + 'Probability:' + str(probs)
    
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

   # output = round(prediction[0], 2)

  # return render_template('index.html', prediction_text="Predicted answer: \n {}".format(s), Your_Question="{}".format(question), Context_Information="{}".format(paragraph) ) 
    # return render_template('index.html', prediction_text="Predicted answer: \n {}".format(answer), prediction_score="Predicted Probability: \n {}".format(round(probs,3)), context="{}".format(paragraph), question="{}".format(question))
    return json.dumps({"status":"OK","prediction_text": answer, "prediction_score": round(probs,3)})
#, render_template('index.html', Context_Information="{}".format(paragraph) ), render_template('index.html', Your_Question="{}".format(question) )


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    feature=list(data.values())
    #prediction = model.predict([np.array(list(data.values()))])
    #int_features = [int(x) for x in request.form.values()]
    id='10000'
    paragraph=features[0]
    questions=features[1]
    model_name=features[2]
    answer=function_main(id, paragraph, question)
        
    
    return jsonify(answer)

if __name__ == "__main__":
    app.run(debug=True)
