import pickle


from transformers import EsmModel, EsmTokenizer
import torch
import transformers
from pathlib import Path
import os
import time


print(transformers.__version__)
model_name = "facebook/esm2_t36_3B_UR50D"
cache_dir = Path(os.getenv("PROJECT", Path.cwd())) / "cache"


tokenizer = EsmTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = EsmModel.from_pretrained(model_name, cache_dir=cache_dir).cuda()

def write2File(_filename, _cont):
    with open(_filename, "w") as f:
        f.writelines(_cont)
        f.close()
    return
def generate_embedding(protein_sequence):

    # Tokenize the input sequence
    inputs = tokenizer(protein_sequence, return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = pretrained_model(**inputs)  # Correct usage with unpacked dict
        token_embeddings = outputs.last_hidden_state  # shape: (1, L+2, D)
    final_embedding = token_embeddings[0, 1:-1]  # shape: (L, D)

    return  final_embedding


def get_esm_feat(_esm_file):
    with open(_esm_file, "rb") as f:
        esm_feat = pickle.load(f)
    # print(_esm_file.shape)
    return esm_feat.cpu().detach().numpy()


t_start = time.time()
#example
fasta_seqs = ["STANHFNAYKLTRPYVAYCADC",
            "MKTFFVLLLAGAGAG",
            "MASQDVKIVVLGGLG",
            "MVHLTPEEKSAVTALWGKV"]
total = 0
total_length = 0
for _ in range(150):
    #print('new iter')
    for seq in fasta_seqs:
        output_file=f"tmp/example_embeddings_{total}.npz"
        #
        # fasta_seq = sys.argv[1]
        # output_file= sys.argv[2]
        result = generate_embedding(seq)

        total_length += len(result)
        #


        with open(output_file, "wb") as f:
            pickle.dump( result,f)
        f.close()


t_end = time.time()
print('time', (t_end-t_start))
print(total_length)
print('tok/s',total_length / (t_end-t_start))
#code to load the files
#embeddings = get_esm_feat(_file)
