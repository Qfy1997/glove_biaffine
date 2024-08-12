

def load(path, max_len=128):
        corpus = []
        with open(path, encoding='utf8') as f:
            for line in f.readlines():
                if line == '\n' or line.startswith('#'):
                    continue
                line = line.split('\t')
                token = line[1]
                corpus.append(token+' ')
        return corpus

if __name__=='__main__':
    file_path = './data/train.conll'
    corpus = load(file_path)
    print(len(corpus))
    word=set(corpus)
    print(len(word))
    with open("./data/train_corpus", "w") as text_file:
        for item in corpus:
            text_file.write(item)
        # text_file.write(corpus)