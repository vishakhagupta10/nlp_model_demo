import torch
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
def function_biomed_roberta(question, answer_text, model_r, tokenizer_r):
    model = model_r
    tokenizer = tokenizer_r
    input_ids = tokenizer.encode(question, answer_text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    
    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1
    
    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a
    
    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)
    start_scores, end_scores = model(torch.tensor([input_ids])) # The tokens representing our input text.
    # token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    
    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end+1])
    answer = tokens[answer_start]
    
    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
      
      # If it's a subword token, then recombine it with the previous token.
      if tokens[i][0:2] == '##':
       answer += tokens[i][2:]
      
      # Otherwise, add a space then the token.
      else:
       answer += ' ' + tokens[i]
    token_labels = []
    for (i, token) in enumerate(tokens):
      token_labels.append('{:} - {:>2}'.format(token, i))
    s_scores = start_scores.detach().tolist()[0]
    e_scores = end_scores.detach().tolist()[0]
    return answer, s_scores, e_scores, token_labels