import numpy as np
import os
import heapq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 

def cos_similarity(batch_emb, dim=0):
    '''
    Return cosine similarity between all sequences of a batch_embedding
    '''
    cos_sims = []
    for i in range(batch_emb.shape[0]):
        for j in range(i+1, batch_emb.shape[0]):     
            X = batch_emb[i]
            Y = batch_emb[j]
            similarity = F.cosine_similarity(X, Y, dim=dim)
            heapq.heappush(cos_sims, (-1*torch.sum(similarity).item(), (i,j)))
    
    return cos_sims

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
            
            # Get cosine similarity of embeddings
            input_embs = model.bert_enc.embeddings(input_ids=input_ids)
            cos_sims = cos_similarity(input_embs)            
            topk = topk_embedding_sentences(cos_sims, 4, input_ids, tokenizer)
            for k in topk.keys():
                print(k)

            input_embs = input_embs.to('cpu')
            bert_embs = model.bert_enc.get_input_embeddings()

            # Plot T-SNE visualization
            X_embedded = TSNE(n_components=2).fit_transform(input_embs[0])
            print(X_embedded.shape)
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.4)

        i += 1
        if i == 3:
            break
    plt.title('T-Sne on BERT Embeddings')
    plt.savefig('embeddings_scattered', dpi=250)

def topk_embedding_sentences(cossims, k, batch_ids, tokenizer):
    top = {}
    for _ in range(k):
        sim, idxs = heapq.heappop(cossims)
        i, j = idxs
        sentence_i = get_sentence(batch_ids[i], tokenizer)
        sentence_j = get_sentence(batch_ids[j], tokenizer)
        top[sentence_i] = 1
        top[sentence_j] = 1
    
    return top

def get_sentence(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    sentence = tokenizer.convert_tokens_to_string(tokens)
    sentence.replace('[PAD]', '')
    return sentence