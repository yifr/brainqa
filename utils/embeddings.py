import numpy as np
import os
import torch

from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 

def emb_visualizer(model, dataset, tokenizer, args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print(args.eval_batch_size)
    fig = plt.figure(figsize=(16,8))
    i = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            input_ids = batch[0]
            print(input_ids.shape)
            
            # Get embeddings
            input_embs = model.bert_enc.embeddings(input_ids=input_ids).cpu()
            bert_embs = model.bert_enc.get_input_embeddings()
            X_embedded = TSNE(n_components=2).fit_transform(input_embs[0])
            plt.scatter(X_embedded[0], X_embedded[1])
        i += 1
        if i == 3:
            break
    plt.title('T-Sne on BERT Embeddings')
    plt.savefig('embeddings_scattered', dpi=250)