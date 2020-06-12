from flask import Flask, Markup, render_template, redirect, url_for, session, request, jsonify
from model_visualize_all import function_main
from model_visualize_all_bert import function_main_bert
from transformers import BertForQuestionAnswering
from transformers import RobertaForQuestionAnswering, RobertaTokenizer 
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from model_clinical_bert import function_clinical
from trans1 import eval
#from model_xlnet import function_xlnet
from model_roberta import function_roberta
from model_biomed_roberta import function_biomed_roberta


from flask_cors import CORS
import torch
import numpy as np
from pytorch_transformers import BertTokenizer, BertForMaskedLM, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()

#tokenizerc = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#modelc = AutoModelForQuestionAnswering.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#tokenizer_rbase = RobertaTokenizer.from_pretrained('roberta-base')
#model_rbase = RobertaForQuestionAnswering.from_pretrained('roberta-base')

#tokenizer_r = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
#model_r = AutoModelForQuestionAnswering.from_pretrained("allenai/biomed_roberta_base")

#from model import function_main
app = Flask(__name__)
CORS(app)


#from flask import Flask, request, jsonify, render_template

g_labels=[]
g_values1=[]
g_values2=[]
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/question', methods=['GET','POST'])
def question():
     if(request.method=='POST'):
        id=""
        paragraph=""
        question=""
        model_name=""
        global g_labels
        global g_values1
        global g_values2
        # int_features = [str(x) for x in request.form.values()]
        #print(len(request.form.values()))
        id="10000"
        # paragraph=int_features[0]
        # question=int_features[1]
       # print(request.form)
        paragraph=request.form["Context_Information"]
        question=request.form["Your_Question"]
        answer=""
        probs=0.0000
        start_score=[]
        end_score=[]
        word_list=[]
        if(request.form["model"]=="bert"):
            answer, probs, start_score, end_score, word_list=function_main_bert(id, paragraph, question)
    #    elif(request.form["model"]=="clinicalbert"):
     #       answer, start_score, end_score, word_list=function_clinical(question, paragraph, modelc, tokenizerc)
     #   elif(request.form["model"]=="roberta"):
     #       answer, start_score, end_score, word_list=function_roberta(question, paragraph, model_rbase, tokenizer_rbase)   
        
      #  elif(request.form["model"]=="biomedroberta"):
      #      answer, start_score, end_score, word_list=function_biomed_roberta(question, paragraph, model_r, tokenizer_r)    
       # elif(request.form["model"]=="biobert"):
        else:
            answer, probs, start_score, end_score, word_list=function_main(id, paragraph, question)
   
    #    answer, probs, start_score, end_score, word_list=function_main(id, paragraph, question)
           # model_name=int_features[2]
           # answer, probs=function_main(id, paragraph, question)
        s="Predicted Answer: " + answer
        t="Predicted Score: " + str(round(probs,3))
        start_values=[]
        end_values=[]
        bar_labels=[]
        token_labels = []
        for (i, token) in enumerate(word_list):
            token_labels.append('{:} - {:>2}'.format(token, i))
        length=len(token_labels)
        bar_labels=token_labels[0:length]
        start_values=start_score[0:length]
        end_values=end_score[0:length]
        g_labels=bar_labels
        g_values1=start_values
        g_values2=end_values
       # app.logger.info(request.form.get("pred", False))
        if (request.form.get("pred", False)) == "predicts":
        #    return json.dumps({"status":"OK","prediction_text": s, "prediction_score": t})
            return render_template("index1.html", context = paragraph, question_c = question, prediction_text = s, prediction_score = t)
            #return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
        #if request.form["visualize"] == "visualize":
        else:
            return redirect(url_for('do_visualize'))
       # if request.form.validate_on_submit():
       # if request.form["predicts"] == "predicts":
       #     return json.dumps({"status":"OK","prediction_text": s, "prediction_score": t})
      #  if request.form["visualize"] == "visualize":
        
       # session["labels"] = bar_labels
        #session["values1"] = start_values
       # session["values2"] = end_values
        
     return render_template('index1.html')
 
@app.route('/filltheblank')
def filltheblank():
    return render_template('index_fill.html')
   # if(request.method=='POST'):
@app.route('/fillblanksutil', methods=['POST'])
def predict():
    question = request.form.get('text')
    sent = ' [CLS] ' + question + ' [SEP]'
    tokenized_text = tokenizer.tokenize(sent)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]
        attention = outputs[-1]
        
    dim = attention[2][0].shape[-1]*attention[2][0].shape[-1]
    a = attention[2][0].reshape(12, dim)
    b = a.mean(axis=0)
    c = b.reshape(attention[2][0].shape[-1],attention[2][0].shape[-1])
   # masked_index = [i for i, x in enumerate(tokenized_text) if x =='[MASK]']
    masked_index = [i for i, x in enumerate(tokenized_text) if x == '[MASK]']
    str=""
    for j in masked_index:
          predicted_index = torch.argmax(predictions[0, j]).item()
          predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
          avg_wgts = c[j]
          focus = [tokenized_text[i] for i in avg_wgts.argsort().tolist()[::-1] if tokenized_text[i] not in ['[SEP]', '[CLS]', '[MASK]']][:5]
          for f in focus:
              question = question.replace(f, '<font color="blue">'+f+'</font>')
              #str=cs(f, "orchid")
             # question = question.replace(f, '<font color="blue">'+f+'</font>')
            # <h1 style="color:'red';">+f+</h1>
             #question = question.replace(f, '<h1 style="color:red">'+f+'</h1>')
          #''<font color="blue">'+f+ '</font>'
         # cs("here we go", "orchid")
	     # ans =  sentence_orig.replace('____', '<font color="red"><b><i>'+predicted_token+'</i></b></font>')
        
          str=str+ question + '<br />' + "predicted text: " + '<font color="red"><b><i>'+predicted_token+'</i></b></font>' + '<br />' + '<br />'
    return str        

@app.route('/translation')
def translation():
    return render_template('index_translate.html')

@app.route('/translateutil', methods=['POST'])
def translate():
    question = request.form.get('text')
    predicted_text = "" 
    predicted_text = eval(question)
    str= "Translated text: " + '<br />' + '<font color="red"><b><i>'+ predicted_text+'</i></b></font>' + '<br />' 
    return str        

@app.route('/predict123',methods=['POST'])
def predict123():
    if(request.method=='POST'):
        id=""
        paragraph=""
        question=""
        model_name=""
        # int_features = [str(x) for x in request.form.values()]
        #print(len(request.form.values()))
        id="10000"
        # paragraph=int_features[0]
        # question=int_features[1]
       # print(request.form)
        paragraph=request.form["Context_Information"]
        question=request.form["Your_Question"]
        answer=""
        probs=0.0
        start_score=[]
        end_score=[]
        word_list=[]
        answer, probs, start_score, end_score, word_list=function_main(id, paragraph, question)
           # model_name=int_features[2]
           # answer, probs=function_main(id, paragraph, question)
        s="Predicted Answer: " + answer
        t="Predicted Score: " + str(round(probs,3))
        start_values=[]
        end_values=[]
        bar_labels=[]
        token_labels = []
        length=len(word_list)
        bar_labels=word_list[0:length]
        start_values=start_score[0:length]
        end_values=end_score[0:length]
       # if request.form.validate_on_submit():
       # if request.form["predicts"] == "predicts":
       #     return json.dumps({"status":"OK","prediction_text": s, "prediction_score": t})
      #  if request.form["visualize"] == "visualize":
        return redirect(url_for('.do_visualize', labels=g_labels, values1=g_values1, values2=g_values2))
           # s=""
           # s=answer + '\n' + 'Probability:' + str(probs)
        
        #final_features = [np.array(int_features)]
        #prediction = model.predict(final_features)
        
           # output = round(prediction[0], 2)
        
          # return render_template('index.html', prediction_text="Predicted answer: \n {}".format(s), Your_Question="{}".format(question), Context_Information="{}".format(paragraph) ) 
        # return render_template('index.html', prediction_text="Predicted answer: \n {}".format(answer), prediction_score="Predicted Probability: \n {}".format(round(probs,3)), context="{}".format(paragraph), question="{}".format(question))
       # if "predicts" in request.form:
          #  if(request.form["predicts"] == "predicts"):
           # if "predicts" in request.form:
          #  return json.dumps({"status":"OK","prediction_text": s, "prediction_score": t})
           # if "visualize" in request.form:
           # if(request.form["visualize"] == "visualize"):
           # if "visualize" in request.form:
        #return redirect(url_for'bar.html', labels=bar_labels, values1=start_values, values2=end_values)

@app.route('/visualize') 
def do_visualize():
   #labels = session["labels"]
   # labels=session.get("labels", {})
  #  values1=session.get("value1", {})
   # values2=session.get("value2", {})
    #app.logger.info(labels)
    return render_template('bar.html', labels=g_labels, values1=g_values1, values2=g_values2 )
@app.route('/visualize123', methods=['POST'])
def visualize():
    #paragraph="Balanced t(11;15)(q23;q15) in a TP53+/+ breast cancer patient from a Li-Fraumeni syndrome family. Li-Fraumeni Syndrome (LFS) is characterized by early-onset carcinogenesis involving multiple tumor types and shows autosomal dominant inheritance. Approximately 70% of LFS cases are due to germline mutations in the TP53 gene on chromosome 17p13.1. Mutations have also been found in the CHEK2 gene on chromosome 22q11, and others have been mapped to chromosome 11q23. While characterizing an LFS family with a documented defect in TP53, we found one family member who developed bilateral breast cancer at age 37 yet was homozygous for wild-type TP53. Her mother also developed early-onset primary bilateral breast cancer, and a sister had unilateral breast cancer and a soft tissue sarcoma. Cytogenetic analysis using fluorescence in situ hybridization of a primary skin fibroblast cell line revealed that the patient had a novel balanced reciprocal translocation between the long arms of chromosomes 11 and 15: t(11;15)(q23;q15). This translocation was not present in a primary skin fibroblast cell line from a brother with neuroblastoma, who was heterozygous for the TP53 mutation. There was no evidence of acute lymphoblastic leukemia in either the patient or her mother, although a nephew did develop leukemia and died in childhood. These data may implicate the region at breakpoint 11q23 and/or 15q15 as playing a significant role in predisposition to breast cancer development."
    #question="What is the inheritance pattern of Li\u2013Fraumeni syndrome?"
# model_name=int_features[2]
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
    answer, probs, start_score, end_score, word_list=function_main(id, paragraph, question)
    s="Predicted Answer: " + answer
    t="Predicted Score: " + str(round(probs,3))
    start_values=[]
    end_values=[]
    bar_labels=[]
    token_labels = []
    for (i, token) in enumerate(word_list):
        token_labels.append('{:} - {:>2}'.format(token, i))
    length=len(token_labels)
    bar_labels=token_labels[0:length]
    start_values=start_score[0:length]
    end_values=end_score[0:length]
    return render_template('bar.html', labels=bar_labels, values1=start_values, values2=end_values)


if __name__ == "__main__":
    app.run(debug=False)
