# Fine Tuning Clinical-Bert for Question Answering System on BIO-ASQ(factoid) dataset



## Overview
Trained upto 5 epoch. 

## Steps to Set-Up

 1. Clone this repository.
 2. Enter into the directory

 ```
 cd bio_bert_ques_ans
 ```
 3. Install requirements using the following command
 ```
 pip install -r requirements.txt
 ```
 4 load pretrained biobert
 ```
!wget -O scibert_uncased.tar https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz
!tar -xvf scibert_uncased.tar
 ```
 5 to convert tf checkpoints to pytorch
 ```
!python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=biobert_v1.1_pubmed/model.ckpt-1000000 --bert_config_file=biobert_v1.1_pubmed/bert_config.json  --pytorch_dump_path=biobert_v1.1_pubmed/pytorch_model.bin
 ```
  
 5. Run the ASQfac.py file for fintuning
 ```
 !python3 ASQfac.py \
  --bert_config_file=biobert_v1.1_pubmed/bert_config.json \
  --vocab_file=biobert_v1.1_pubmed/vocab.txt \
  --train_file=BioASQ-train-factoid-4b.json \
  --predict_file=ASQdev.json \
  --init_checkpoint=biobert_v1.1_pubmed/pytorch_model.bin\
  --save_checkpoints_steps=3000  \
  --train_batch_size=12  \
  --num_train_epochs=5
 ```
 
## Result

F1 score

 ```
87.92
 ```


## Conclusion

For complete result refer .ipynb file of this folder




