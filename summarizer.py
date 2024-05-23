
from transformers import AutoTokenizer, AutoModelForCausalLM

import math
import sys



from huggingface_hub import login

login(token="<enter your token here>")

def cache_model(cache_dir = "/working/"):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", cache_dir= cache_dir)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", pad_token_id = tokenizer.eos_token_id,cache_dir= cache_dir, device_map="auto")



def _summarize(text, model, tokenizer):
    B_INST, E_INST = "[INST] ", " [/INST]"
    prompt = f"{B_INST}Write a concise summary of the following text:\n\n[TEXT_START]\n\n{text}\n\n[TEXT_END]\n\n{E_INST}"

   
    inputs = tokenizer( prompt,return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1000, use_cache=True, do_sample=True,temperature=0.2, top_p=0.95)

    prompt_length = inputs['input_ids'].shape[1]

    #print(f"creating a summary for {len(inputs.sequence_ids())} tokens")
    
    summary = tokenizer.decode(outputs[0][prompt_length:-1])
    
    return summary

# chunk tokens into list of texts
def chunk(tokens, max_token_length,tokenizer):
    token_length = len(tokens)
    k = math.ceil(token_length /max_token_length)
    chunk_sizes = [token_length // k + (1 if x < token_length % k else 0)  for x in range (k)]
    #print(token_length, k, chunk_sizes)
    last = 1
    texts =[]
    for l in chunk_sizes:
        sub_sequence_ids = tokens[last: last+l]
        last +=l
        texts.append(tokenizer.decode(sub_sequence_ids))
    return texts

def summarize(text, cache_dir = "/working/", tokenizer = None, model=None):
    max_token_length = 1000

    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", model_max_length = max_token_length, cache_dir= cache_dir)
    if model == None:

        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", pad_token_id = tokenizer.eos_token_id,cache_dir= cache_dir, device_map="auto")

    #model.to('cpu')

    model.half()
    tokens = tokenizer.encode( text)#,return_tensors="pt")
    summary = ""
    complete_runthroughs = 0
    while len(tokens) > max_token_length:
        texts = chunk(tokens, max_token_length, tokenizer)
        #text = _summarize(tokens, model,tokenizer)
        
        summaries = []
        for text in texts:
            sub_summary = _summarize(text, model, tokenizer)
            summaries.append(sub_summary)
        summary = " ".join(summaries)
        tokens = tokenizer.encode(summary)
        complete_runthroughs+=1
        print(f"run through entire text doc {complete_runthroughs} times")
    return _summarize(summary, model, tokenizer)


if __name__ == "__main__":


    source = sys.argv[1]

    i = open(source, "r")
    content = i.read()

    i.close()
   
    summary = summarize(content)
   
    dest = sys.argv[2]

    with open(dest,"w") as f:
        f.write(summary)

