import re
import PyPDF2
import os
import random
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
parties = ["PVDA","SGP","Ondernemerspartij","Libertarische Partij","Lokaal in de Kamer","Groenlinks","PVV","VoorNederland","Piratenpartij","DENK","De Burger Beweging","D66","CDA","50PLUS","VVD","Partij voor de Dieren","MenS en Spirit/Basisinkomen Partij/V","ChristenUnie","SP","NIEUWE WEGEN","JEZUS LEEFT","Vrijzinnige Partij","Forum voor Democratie"]


def load_data(parties_and_sentences,getter):
    train_x = list()
    train_y = list()
    for party_index,name in enumerate(parties_and_sentences.keys()):
        print(name + " has "  + str(len(parties_and_sentences[name])) + " sentences ")
        for sentence in parties_and_sentences[name]:
            numeric_sentence = list()
            for word in sentence:
                numeric_sentence.append(getter.get_id_of_word(word))
            train_x.append(numeric_sentence)
            train_y.append(party_index)
    train_set = list(zip(train_x,train_y))
    random.shuffle(train_set)
    percentage_split = 0.8
    train = train_set[:int(percentage_split*len(train_set))]
    test = train_set[int(percentage_split*len(train_set)):]
    return zip(*train),zip(*test)


class IdOfWordGetter:
    def get_word_of_id(self, index):
        return self.vocab[index]

    def get_id_of_word(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return self.word_dict[_UNK]

    def __init__(self, vocab):
        self.vocab = vocab
        self.word_dict = dict()
        for index, word in enumerate(vocab):
            self.word_dict[word] = index

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def get_index_end_sentence(tokens):
    for index, token in enumerate(tokens):
        if token in ["?", "!", "."]:
            return index
    return len(tokens)


def get_sentences_from_tokens(tokens):
    sentences = []
    while len(tokens) > 0:
        nextindex = get_index_end_sentence(tokens)
        if nextindex == len(tokens):
            sentences.append(tokens)
            return sentences
        else:
            sentences.append(tokens[:nextindex + 1])
            tokens = tokens[nextindex + 1:]
    return sentences


def get_parties_and_sentences(partypath):
    partijprogrammas = os.listdir(partypath)
    part_sentences = dict()

    for partijprogramma_name in partijprogrammas:
        print(partijprogramma_name)
        pdf_obj = open(os.path.join(partypath, partijprogramma_name), 'rb')
        pdfreader = PyPDF2.PdfFileReader(pdf_obj)

        sentences = []

        for page in pdfreader.pages:
            text = page.extractText()
            # remove all words with a number
            text = text.replace("\n", " ")

            text = re.sub(r'\w*\d\w*', '', text).strip()

            # make lower case
            text = text.lower()
            if (len(text) > 0):
                tokens = basic_tokenizer(text)
                sentences_this_page = get_sentences_from_tokens(tokens)
                sentences.extend(sentences_this_page)
        part_sentences[partijprogramma_name] = sentences
    return part_sentences

def vocab_from_sentences(parties_and_sentences):
    vocab = dict()

    for party in parties_and_sentences.keys():
        for line in parties_and_sentences[party]:
            for word in line:
                if not word in vocab:
                    vocab[word] = 0
                vocab[word] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    sorted_vocab = [word for word in sorted_vocab if vocab[word] >= 5]
    vocab_list = _START_VOCAB + sorted_vocab

    return vocab_list