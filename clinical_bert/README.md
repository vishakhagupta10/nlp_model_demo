# Fine Tuning Clinical-Bert for Question Answering System on BIO-ASQ(factoid) dataset



## Overview
Trained upto 5 epoch. 

## Steps to Set-Up

 1. Clone this repository.
 2. Enter into the directory

 ```
 cd clinical_bert
 ```
 3. Install requirements using the following command
 ```
 pip install -r requirements.txt
 ```
 4. Run the clinical_bert.py file for fintuning
 ```
 !python clinical_bert.py \
    --model_type BIO_ClinicalBERT  \
    --model_name_or_path BIO_ClinicalBERT \
    --do_train \
    --do_eval \
    --train_file BioASQ-train-factoid-4b.json\
    --predict_file  ASQdev.json \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir output/ \
    --per_gpu_eval_batch_size=16  \
    --per_gpu_train_batch_size=16   \
    --save_steps 5000
 ```
 
## Result

F1 score

 ```
79.68
 ```

Exact match

 ```
77.21
 ```
## Conclusion

For complete result refer .ipynb file of this folder





