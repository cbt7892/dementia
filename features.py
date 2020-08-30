import nltk
import numpy as np
import math
from scipy import spatial
from nltk.corpus import brown
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()


# POS TAGS
# NN	noun, singular 'desk', #NNS	noun plural	'desks', #NNP	proper noun, singular	'Harrison', #NNPS	proper noun, plural	'Americans'
# PRP	personal pronoun	I, he, she, #PRP$	possessive pronoun	my, his, hers,
# VB	verb, base form	take, #VBD	verb, past tense	took, #VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken, #VBP	verb, sing. present, non-3d	take, #VBZ	verb, 3rd person sing. present	takes
# RB	adverb	very, silently,#RBR	adverb, comparative	better, #RBS	adverb, superlative	best
# CC	coordinating conjunction, #IN	preposition/subordinating conjunction
# TO	to	go 'to' the store, #RP	particle	give up, #MD	modal	could, will
# CD	cardinal digit, #LS	list marker	1), #FW	foreign word, #UH	interjection	errrrrrrrm
# DT	determiner, #PDT	predeterminer	'all the kids'
# EX	existential there (like: "there is" ... think of it like "there exists")
# JJ	adjective	'big', #JJR	adjective, comparative	'bigger', #JJS	adjective, superlative	'biggest'
# POS	possessive ending	parent's
# WDT	wh-determiner	which, #WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose, #WRB	wh-abverb	where, when

def similarity(content, POS_tag):
    temp_info = nltk.pos_tag(nltk.word_tokenize(content))
    temp_fd = nltk.FreqDist(tag for (word, tag) in temp_info)
    tot_pos = sum([temp_fd[tag] for tag in POS_tag])

    local_pos_vec = []
    for tag in POS_tag:
        if tag in list(temp_fd.keys()):
            local_pos_vec.append(temp_fd[tag] / tot_pos)
        else:
            local_pos_vec.append(0)

    return local_pos_vec


def get_tag_info(data):
    # ----------------- Initialize ---------------
    data_tag_info = []
    # --------------- Noun tag and Verb tag lists ---------------

    noun_list = ['NN', 'NNS', 'NNP', 'NNPS']
    verb_list = ['VB', 'VBD', 'VBG', 'VBN', 'VBP']

    # ---------- Define production rules / VP, NP definition ----------
    grammar = r"""
    DTR: {<DT><DT>}
    NP: {<DT>?<JJ>*<NN.*>} 
    PP: {<IN><NP>} 
    VPG: {<VBG><NP | PP>}
    VP: {<V.*><NP | PP>}     
    CLAUSE: {<NP><VP>} 
    """

    # ----------- Number of concepts mentioned ------------------------

    # SUBJECTS (boy, girl, mother), # PLACES (kitchen, exterior seen through the window),
    # OBJECTS (cabinet, cookies, counter, curtain, dishes on the counter, faucet, floor, jar, plate, sink, stool, water, window) and
    # ACTIONS (boy taking the cookie, boy or stool falling, woman drying or washing dishes/plate,
    # water overflowing or spilling, the girl asking for a cookie,
    # woman unconcerned by the overflowing, # woman indifferent to the children)

    cookie_pic_list = ['cookie', 'jar', 'stool', 'steal',
                       'sink', 'kitchen',
                       'window', 'curtain', 'fall']
    list1 = ['mother', 'woman', 'lady']
    list2 = ['girl', 'daughter', 'sister']
    list3 = ['boy', 'son', 'child', 'kid', 'brother']
    list4 = ['dish', 'plate', 'cup']
    list5 = ['overflow', 'spill', 'running']
    list6 = ['dry', 'wash']
    list7 = ['faucet']
    list8 = ['counter', 'cabinet']
    list9 = ['water']

    # ---------- Distribution feature ----------
    text = brown.words(categories='news')
    tag_info = nltk.pos_tag(text)
    tag_fd = nltk.FreqDist(tag for (word, tag) in tag_info)
    del_key = []
    for key in tag_fd.keys():
        if not key.isalpha():
            del_key.append(key)
    while not (del_key == []):
        tag_fd.pop(del_key.pop(), None)

    POS_tag = ['NN', 'IN', 'DT', 'VBD', 'VBFG', 'VBG', 'PRP', 'JJ', 'NNP', 'RB', 'NNS', 'CC']
    tot_pos = sum([tag_fd[tag] for tag in POS_tag])

    global_pos_vec = []
    for tag in POS_tag:
        if tag in list(tag_fd.keys()):
            global_pos_vec.append(tag_fd[tag] / tot_pos)
        else:
            global_pos_vec.append(0)

    # ---------------------tagging information -------------------
    content = data
    text = nltk.word_tokenize(content)

    # ========= LEXICOSYNTACTIC FEATURES =========

    #  ------- POS tagging -------
    tag_info = np.array(nltk.pos_tag(text))
    tag_fd = nltk.FreqDist(tag for i, (word, tag) in enumerate(tag_info))
    freq_tag = tag_fd.most_common()
    data_tag_info.append(freq_tag)

    # ------- Lemmatize each word -------
    text_root = [lmtzr.lemmatize(j) for indexj, j in enumerate(text)]
    for indexj, j in enumerate(text):
        if tag_info[indexj, 1] in noun_list:
            text_root[indexj] = lmtzr.lemmatize(j)
        elif tag_info[indexj, 1] in verb_list:
            text_root[indexj] = lmtzr.lemmatize(j, 'v')

    # ------- Phrase type -------
    sentence = nltk.pos_tag(text)
    cp = nltk.RegexpParser(grammar)
    phrase_type = cp.parse(sentence)

    # ------- Pronoun frequency -------
    prp_count = sum([pos[1] for pos in freq_tag if pos[0] == 'PRP' or pos[0] == 'PRP$'])

    # ------- Noun frequency -------
    noun_count = sum([pos[1] for pos in freq_tag if pos[0] in noun_list])

    # ------- Gerund frequency -------
    vg_count = sum([pos[1] for pos in freq_tag if pos[0] == 'VBG'])

    # ------- Pronoun-to-Noun ratio -------
    if noun_count != 0:
        prp_noun_ratio = prp_count / noun_count
    else:
        prp_noun_ratio = prp_count

    # Noun phrase, Verb phrase, Verb gerund phrase frequency
    NP_count = 0
    VP_count = 0
    VGP_count = 0
    for index_t, t in enumerate(phrase_type):
        if not isinstance(phrase_type[index_t], tuple):
            if phrase_type[index_t].label() == 'NP':
                NP_count = NP_count + 1
            elif phrase_type[index_t].label() == 'VP':
                VP_count = VP_count + 1
            elif phrase_type[index_t].label() == 'VGP':
                VGP_count = VGP_count + 1

    # ------- TTR type-to-token ratio -------
    numtokens = len(text)
    freq_token_type = Counter(text)  # or len(set(text)) # text_root
    v = len(freq_token_type)
    ttr = float(v) / numtokens

    # ------- Honore's statistic -------
    freq_token_root = Counter(text_root)
    occur_once = 0
    for j in freq_token_root:
        if freq_token_root[j] == 1:
            occur_once = occur_once + 1
    v1 = occur_once
    R = 100 * math.log(numtokens / (1 - (v1 / v)))

    # ------- Automated readability index -------
    num_char = len([c for c in content if c.isdigit() or c.isalpha()])
    num_words = len([word for word in content.split(' ') if not word == '' and not word == '.'])
    num_sentences = content.count('.') + content.count('?')
    ARI = 4.71 * (num_char / num_words) + 0.5 * (num_words / num_sentences) - 21.43

    # ------- Coleman–Liau index -------
    L = (num_char / num_words) * 100
    S = (num_sentences / num_words) * 100
    CLI = 0.0588 * L - 0.296 * S - 15.8

    # ------- Word-to-sentence_ratio -------
    word_sentence_ratio = num_words / num_sentences

    # ------- Mean Length Utterance (MLU) -------
    # NEEDS TO BE IMPLEMENTED TO INCLUDE PAUSES
    num_utterances = content.count('.') + content.count('?')
    MeanLenUtter = num_words / num_utterances

    # ========= SEMANTIC FEATURES =========

    # ------- Mention of key concepts -------
    num_concepts_mentioned = len(set(cookie_pic_list) & set(freq_token_root)) \
                             + len(set(list1) & set(freq_token_root)) + len(set(list2) & set(freq_token_root)) \
                             + len(set(list3) & set(freq_token_root)) + len(set(list4) & set(freq_token_root)) \
                             + len(set(list5) & set(freq_token_root)) + len(set(list6) & set(freq_token_root)) \
                             + len(set(list7) & set(freq_token_root)) + len(set(list8) & set(freq_token_root)) \
                             + len(set(list9) & set(freq_token_root))

    # ========= ACOUSTIC FEATURES =========

    # ------- Pauses and unintelligible count -------
    count_pauses = 0

    count_unintelligible = 0 # NEEDS TO BE IMPLEMENTED

    count_trailing = 0

    count_repetitions = 0

    # ---------- Distribution feature ----------
    local_pos_vec = similarity(content, POS_tag)
    sim_score = 1 - spatial.distance.cosine(local_pos_vec, global_pos_vec)

    # ---------- Bruten Index ----------
    freq_lemmtoken_type = Counter(text_root)
    vl = len(freq_lemmtoken_type)

    bruten = float(vl) ** (numtokens ** -0.0165)

    feature_set = [ttr, R, num_concepts_mentioned, ARI, CLI,
                   prp_count, prp_noun_ratio, vg_count, NP_count, VP_count,
                   word_sentence_ratio, MeanLenUtter, count_pauses, count_unintelligible,
                   count_trailing, count_repetitions, sim_score, bruten]

    return data_tag_info, feature_set


def main():
    speech = ""
    print(get_tag_info(speech)[1])


if __name__ == '__main__':
    main()
