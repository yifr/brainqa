import numpy as np
import os
import heapq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 
import matplotlib.text as mtext

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

class WrapText(mtext.Text):
    def __init__(self,
                 x=0, y=0, text='',
                 width=0,
                 **kwargs):
        mtext.Text.__init__(self,
                 x=x, y=y, text=text,
                 wrap=True,
                 **kwargs)
        self.width = width  # in screen pixels. You could do scaling first

    def _get_wrap_line_width(self):
        return self.width

def emb_visualizer(model, dataset, tokenizer, args, embed_vis=True, latent_vis=False):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print(args.eval_batch_size)

    fig = plt.figure(figsize=(16, 8))
    title = 'cos_sim_embs_latents'

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

            # VQVAE Reconstruction visualization
            outputs_VQVAE = model.vqvae_model(input_embs) 
            vq_embedding_loss, embeds_reconstructed, vqvae_ppl, vqvae_latent_states = outputs_VQVAE   

            if latent_vis:
                idx = 5
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
            
                title = 'latent_reconstruction_d%02d' % idx
                input_embs = input_embs.to('cpu')
                embeds_reconstructed = embeds_reconstructed.to('cpu')
                sentence = get_sentence(input_ids[idx], tokenizer)
                wtxt = WrapText(0.2, 0.9, sentence, width=1200, fontsize=12, ha='left', va='top')
                ax2.add_artist(wtxt)

                ax2.legend()

                print('Sentence: ', sentence)
                bert_embedded = TSNE(n_components=2, init='pca', random_state=42).fit_transform(input_embs[idx])
                vqvae_reconstructions = TSNE(n_components=2, init='pca', random_state=42).fit_transform(embeds_reconstructed[idx])
                ax1.scatter(bert_embedded[:, 0], bert_embedded[:, 1], alpha=0.4, label='BERT Embeddings')
                ax1.scatter(vqvae_reconstructions[:, 0], vqvae_reconstructions[:, 1], alpha=0.4, label='Reconstructed Embeddings')
                ax1.legend()

            elif embed_vis: 
                cos_sims = cos_similarity(input_embs)            
                topk = topk_embedding_sentences(cos_sims, 1, input_ids, tokenizer)
                bottomk = topk_embedding_sentences(cos_sims, 1, input_ids, tokenizer, bottom=True)

                input_embs = input_embs.to('cpu')

                sim_embs = fig.add_subplot(241)
                sim_latents = fig.add_subplot(242)
                sim_text_a = fig.add_subplot(243)
                sim_text_b = fig.add_subplot(244)

                diff_embs = fig.add_subplot(245)
                diff_latents = fig.add_subplot(246)
                diff_text_a = fig.add_subplot(247)
                diff_text_b = fig.add_subplot(248)

                sim_embs.title.set_text('Most similar embeddings')
                sim_latents.title.set_text('Corresponding VQVAE latent states')
                sim_text_a.title.set_text('Corresponding input text (Passage 1)')
                sim_text_b.title.set_text('Corresponding input text (Passage 2)')

                diff_embs.title.set_text('Least similar embeddings')
                diff_latents.title.set_text('Corresponding VQVAE latent states')
                diff_text_a.title.set_text('Corresponding input text (Passage 1)')
                diff_text_b.title.set_text('Corresponding input text (Passage 2)')

                for i, k in enumerate(topk.keys()):
                    legend_text = 'INDEX: ' + str(k)  + ', ' + str(topk[k][0]) + ': ' + topk[k][1]
                    print(legend_text)
                    
                    # Plot T-SNE visualization
                    # Embeddings:
                    embs_embedded = TSNE(n_components=2, init='pca', random_state=42).fit_transform(input_embs[k])
                    sim_embs.scatter(embs_embedded[:, 0], embs_embedded[:, 1], alpha=0.4, label='Passage #%d'%(i+1))

                    # Latents
                    latents = vqvae_latent_states[k].to('cpu')
                    latents_embedded = TSNE(n_components=2, init='pca', random_state=42).fit_transform(latents)
                    sim_latents.scatter(latents_embedded[:, 0], latents_embedded[:, 1], alpha=0.4, label='Passage #%d'%(i+1))

                    # Sentence
                    wtxt = WrapText(0.01, 0.99, topk[k][1], width=1200, fontsize=8, ha='left', va='top')
                    if i == 1:
                        sim_text_a.add_artist(wtxt)
                    else:
                        sim_text_b.add_artist(wtxt)

                sim_embs.legend(prop={'size': 6})
                sim_latents.legend(prop={'size': 6})

                print('LEAST SIMILAR')
                for i, k in enumerate(bottomk.keys()):
                    # Plot T-SNE visualization
                    # Embeddings:
                    embs_embedded = TSNE(n_components=2, init='pca', random_state=42).fit_transform(input_embs[k])
                    diff_embs.scatter(embs_embedded[:, 0], embs_embedded[:, 1], alpha=0.4, label='Passage #%d'%(i+1))

                    # Latents
                    latents = vqvae_latent_states[k].to('cpu')
                    latents_embedded = TSNE(n_components=2, init='pca', random_state=42).fit_transform(latents)
                    diff_latents.scatter(latents_embedded[:, 0], latents_embedded[:, 1], alpha=0.4, label='Passage #%d'%(i+1))

                    # Sentence
                    wtxt = WrapText(0.01, 0.99, bottomk[k][1], width=1200, fontsize=8, ha='left', va='top')
                    if i == 1:
                        diff_text_a.add_artist(wtxt)
                    else:
                        diff_text_b.add_artist(wtxt)

                diff_embs.legend(prop={'size': 6})
                diff_latents.legend(prop={'size': 6})
        break

    plt.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.title('Comparing BERT Embeddings and their VQVAE Reconstructions')
    plt.savefig('images/'+title, bbox_inches='tight', dpi=350)

def topk_embedding_sentences(cossims, k, batch_ids, tokenizer, bottom=False):
    '''
    Returns the k most or least similar embeddings
    '''
    top = {}
    targets = None
    if not bottom:
        targets = heapq.nsmallest(k, cossims)
    else:
        # Least similar embeddings have greatest values in minheap
        targets = heapq.nlargest(k, cossims)

    for sim, idxs in targets:
        i, j = idxs
        sentence_i = get_sentence(batch_ids[i], tokenizer)
        sentence_j = get_sentence(batch_ids[j], tokenizer)
        top[i] = (sim, sentence_i)
        top[j] = (sim, sentence_j)

    return top

def get_sentence(ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    sentence = tokenizer.convert_tokens_to_string(tokens)
    sentence = sentence.replace('[PAD]', '')
    return sentence