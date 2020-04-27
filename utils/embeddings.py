def emb_visualizer(model, data):
    train_sampler = RandomSampler(data)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8)
    
def next_question(dataloader):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    return examples