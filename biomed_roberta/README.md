# Fine Tuning BIOMED-RoBERTa for Question Answering System on BIO-ASQ(factoid) dataset



## Overview
Trained upto 5 epoch. 

## Steps to Set-Up

 1. Clone this repository.
 2. Enter into the directory

 ```
 cd biomed_roberta
 ```
 3. Install requirements using the following command
 ```
 pip install -r requirements.txt
 ```
 4. Run the biomed_roberta_try.py file for fintuning
 ```
 !python biomed_roberta_try.py \
    --model_type biomed_roberta_base  \
    --model_name_or_path biomed_roberta_base \
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
83.44
 ```

Exact match

 ```
80.86
 ```
## Conclusion

For complete result refer .ipynb file of this folder





