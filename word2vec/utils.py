import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from vocab import Vocab

# Constants
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOW_TOKEN = "<bow>"
EOW_TOKEN = "<eow>"

WEIGHT_INIT_RANGE = 0.1
def load_treebank():
    from nltk.corpus import treebank
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_sentence_polarity():
    from nltk.corpus import sentence_polarity

    vocab = Vocab.build(sentence_polarity.sents())

    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data, test_data, vocab

def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = (torch.arange(max_len).expand(lengths.shape[0], max_len)).to(device)< lengths.unsqueeze(1)
    return mask


def load_crawldata():
    import re
    strs = []
    print("1.加载并清理数据")
    with open("../data/crawldataForWord2vec.txt", encoding='utf-8') as f:
        data = f.readlines()
        data = data[:int(len(data) * 0.2)]
    vocab = Vocab()
    corpus=[]
    for str_ in tqdm(data,desc="2.构建词表"):
        if str_ == "\n": continue
        str_ = re.sub(r'\d +', '', str_)
        table = str.maketrans(
            {'[': '', '!': '', '"': '', '#': '', '$': '', '%': '', '{': '', '}': '', '~': '', ']': '', '<': '', '>': '',
             'n': '', '�':''})
        str_ = str_.translate(table)
        str_ = str_.strip()
        str_ = [one for one in str_]
        str_=vocab.convert_tokens_to_ids(str_)
        corpus.append(str_)
    # vocab = Vocab.build(strs, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    # corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in strs]
    return corpus, vocab

def load_reuters():
    from nltk.corpus import reuters
    text = reuters.sents()[:int(len(reuters.sents())*0.2)]
    # lowercase (optional)
    text = [[word.lower() for word in sentence] for sentence in text]
    vocab = Vocab.build(text, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]

    return corpus, vocab

def save_pretrained(vocab, embeds, save_path,total_losses):
    """
    Save pretrained token vectors in a unified format, where the first line
    specifies the `number_of_tokens` and `embedding_dim` followed with all
    token vectors, one token per line.
    """

    with open(save_path, "w",encoding="utf-8") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    with open(save_path.split('_')[0]+'_loss.txt', "w",encoding="utf-8") as writer:
        txt=""
        for idx, loss in enumerate(total_losses):
            txt+=str(loss)+" "
        writer.write(f"{txt}")
    print(f"Pretrained embeddings saved to: {save_path}")

def load_pretrained(load_path):
    with open(load_path, "r") as fin:
        # Optional: depending on the specific format of pretrained vector file
        n, d = map(int, fin.readline().split())
        tokens = []
        embeds = []
        for line in fin:
            line = line.rstrip().split(' ')
            token, embed = line[0], list(map(float, line[1:]))
            tokens.append(token)
            embeds.append(embed)
        vocab = Vocab(tokens)
        embeds = torch.tensor(embeds, dtype=torch.float)
    return vocab, embeds

def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
    return data_loader

def init_weights(model):
    for name, param in model.named_parameters():
        if "embedding" not in name:
            torch.nn.init.uniform_(
                param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE
            )

