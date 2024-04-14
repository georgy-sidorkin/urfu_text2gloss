import pandas as pd
import pymorphy2
import os
import wget
import zipfile
import gensim


def get_list_gloss_dict(file_name="RSL_class_list.txt", sep="\t"):
    df = pd.read_csv(file_name, sep=sep)
    lst = df['1'].to_list()
    return lst


def define_part_of_speech(gloss_list: list):
    dict_with_part = dict()
    # словарь для замены частей речи
    compare_dict = {
        "NPRO": "NOUN",
        "ADJF": "ADJ",
        "ADJS": "ADJ",
        "INFN": "VERB",
        "NUMR": "NUM",
        "ADVB": "ADV",
        "COMP": "ADV",
        "PRED": "ADV",
    }

    for word in gloss_list:
        word = word.replace("ё", "е")
        if len(word.split(" ")) != 1:
            dict_with_part[word] = None
        elif word.isdigit():
            dict_with_part[word] = "NUM"
        else:
            part_of_speech = pymorphy2.MorphAnalyzer(lang='ru').parse(word)[0].tag.POS
            if part_of_speech in compare_dict:
                part_of_speech = compare_dict[part_of_speech]
            dict_with_part[word] = str(part_of_speech)
    return dict_with_part


def download_model_rusvectores(model_id="180"):
    if not os.path.exists("models"):
        os.mkdir("models")
    if os.path.exists(f"models/{model_id}"):
        print("Model is downloaded already")
    else:
        model_url = f'http://vectors.nlpl.eu/repository/11/{model_id}.zip'
        m = wget.download(model_url)
        with zipfile.ZipFile(f'{model_id}.zip', 'r') as zip_ref:
            unzip_path = "models/" + str(model_id)
            zip_ref.extractall(unzip_path)
    return f"models/{model_id}"


def load_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model


def get_word_vector(model, word, part_of_speech):
    return model[f"{word}_{part_of_speech}"]
