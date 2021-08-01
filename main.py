from PIL import Image, ImageDraw, ImageEnhance
import pytesseract
import statistics
import os
from xml.etree import ElementTree as ET
from matplotlib import pyplot as plt
import pyxlsb
from IPython.display import display
from catboost import CatBoostClassifier
from catboost import *

import pickle
import nltk

import pandas as pd
import numpy as np
import cv2
import time
import numpy as np
from catboost import CatBoost, Pool


def train(csv_path):
    # read the dataset
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # train_d = df['Size'][:200000]

    model = CatBoostClassifier()
    model1 = CatBoost
    model2 = CatBoostRegressor

    # TODO init your  X_train, y_train, X_test, y_test however you do it
    # X_train = df.loc[:, df.columns != 'Bolded', df.columns != 'Book Name', df.columns != 'Coordinates',
    #           df.columns != 'Char'][:200000]
    # 	Book Name	Page Number	Char	Bolded	Coordinates	Size	Height	Width	ASCII val	counter	Line Location	Left	Right	Top	Bottom
    data = df[['Page Number', 'Size', 'Height', 'Width', 'ASCII val', 'counter', 'Line Location', 'Left', 'Right', 'Top', 'Bottom']]
    X_train = data[:130000]
    X_test = data[130001:]
    # X_test = df.loc[:, df.columns != 'Bolded', df.columns != 'Book Name', df.columns != 'Coordinates',
    #           df.columns != 'Char'][200001:]
    y_train = df['Bolded'][:130000]

    # train

    model.fit(X_train, y_train)
    # model1.fit(X_train, y_train)
    # model2.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    y_pred = y_pred.tolist()
    print(y_pred)

    actual_bold = df['Bolded'][130001:]
    actual_bold = actual_bold.tolist()
    print("actual bold")
    print(actual_bold)

    print("size of pred: " + str(len(y_pred)))
    print(type(y_pred))
    print("size of actual: " + str(len(actual_bold)))
    print(type(actual_bold))
    same = 0
    diff = 0
    # counter = 0
    print(type(y_pred[3]))
    predict = []
    for i in y_pred:
        if i == 'True':
            predict.append(True)
        elif i == 'False':
            predict.append(False)
    print(len(predict))
    print(type(actual_bold[3]))
    # false_positive = 0
    # false_negative = 0
    for i in range(len(predict)):
        # if i == 0:
        #     continue
        # print(y_pred[i], actual_bold[i])
        if predict[i] != actual_bold[i]:
            # if (predict[i] == True) and (actual_bold[i] == False):
            #     false_positive += 1
            # elif (predict[i] == False) and (actual_bold == True):
            #     false_negative += 1
            diff += 1
        else:
            same += 1
        # counter += 1
    print("same: " + str(same))
    print("diff: " + str(diff))
    print("percentage correct: " + str(float(same/len(predict)) * 100) + "%")
    # print("false positive: " + str(false_positive))
    # print("false negative: " + str(false_negative))
    # print("false positive percent: " + str(float(false_positive/diff) * 100) + "%")

    # predict 1
    # predict
    y_pred1 = model1.predict(X_test)
    y_pred1 = y_pred1.tolist()

    print("size of pred: " + str(len(y_pred)))
    print(type(y_pred))
    print("size of actual: " + str(len(actual_bold)))
    print(type(actual_bold))
    same = 0
    diff = 0
    # counter = 0
    print(type(y_pred[3]))
    predict = []
    for i in y_pred:
        if i == 'True':
            predict.append(True)
        elif i == 'False':
            predict.append(False)
    print(len(predict))
    print(type(actual_bold[3]))
    # false_positive = 0
    # false_negative = 0
    for i in range(len(predict)):
        # if i == 0:
        #     continue
        # print(y_pred[i], actual_bold[i])
        if predict[i] != actual_bold[i]:
            # if (predict[i] == True) and (actual_bold[i] == False):
            #     false_positive += 1
            # elif (predict[i] == False) and (actual_bold == True):
            #     false_negative += 1
            diff += 1
        else:
            same += 1
        # counter += 1
    print("same: " + str(same))
    print("diff: " + str(diff))
    # print("percentage c



    #
    # bolded_binary = []
    # for i in df['Bolded']:
    #     if i == True:
    #         bolded_binary.append(1)
    #     elif i == False:
    #         bolded_binary.append(0)
    # print(bolded_binary)
    # train_l = bolded_binary[:200000]
    # print("data: " + str(train_d))
    # print("labels: " + str(train_l))
    #
    # test_d = df['Size'][200001:]

    # train_data = np.random.randint(0, 100, size=(100, 10))
    # train_labels = np.random.randint(0, 2, size=(100))
    # test_data = np.random.randint(0, 100, size=(50, 10))
    #
    #
    # print("train data: " + str(train_data))
    # print("train labels: " + str(train_labels))
    # print("test data: " + str(test_data))
    #
    # train_pool = Pool(train_d, train_l)
    # # train_pool = Pool(train_data, train_labels)
    # #
    # test_pool = Pool(test_d)
    # # test_pool = Pool(test_data)
    # # # specify training parameters via map
    # #
    # param = {'iterations': 5}
    # model = CatBoost(param)
    # # train the model
    # model.fit(train_pool)
    # # make the prediction using the resulting model
    # preds_class = model.predict(test_pool, prediction_type='Class')
    # preds_proba = model.predict(test_pool, prediction_type='Probability')
    # preds_raw_vals = model.predict(test_pool, prediction_type='RawFormulaVal')
    # print("Class", preds_class)
    # print("Proba", preds_proba)
    # print("Raw", preds_raw_vals)




def ocr():
    print("             GETTING TEXT                  ")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR/tesseract'
    # print("low score")
    # path = r"C:\Users\emwil\Downloads\low_score_images"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print(entry.name)
    #             im = Image.open(entry.path)
    #             text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    #             name = entry.name.removesuffix(".tif")
    #             output_path = r"C:\Users\emwil\Downloads\low_score_text" + os.sep + name + ".txt"
    #             file = open(output_path, "w", encoding='utf-8-sig')
    #             file.write(text)
    #             file.close()

    #             # with open(output_path, 'wb') as f:
    #             #     f.write(text)
    #             # file.close()

    # print("reg images")
    # path = r"C:\Users\emwil\Downloads\processed_images\processed_images"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print(entry.name)
    #             im = Image.open(entry.path)
    #             text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    #             name = entry.name.removesuffix(".tif")
    #             output_path = r"C:\Users\emwil\Downloads\tes_reg_text" + os.sep + name + ".txt"
    #             file = open(output_path, "w", encoding='utf-8-sig')
    #             file.write(text)
    #             file.close()
    #
    # print("cleaned images")
    # path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images_png"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".png") and entry.is_file():
    #             print(entry.name)
    #             try:
    #                 im = Image.open(entry.path)
    #                 text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    #                 name = entry.name.removesuffix(".png")
    #                 output_path = r"C:\Users\emwil\Downloads\tes_cleaned_png_text" + os.sep + name + ".txt"
    #                 file = open(output_path, "w", encoding='utf-8-sig')
    #                 file.write(text)
    #                 file.close()
    #             except:
    #                 continue
    #
    # print(" ")
    # print("cleaned 0 ")
    # path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\0"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print(entry.name)
    #             try:
    #                 # im = Image.open(entry.path)
    #                 text = pytesseract.image_to_string(entry.path, lang="heb_tess_dict")
    #                 name = entry.name.removesuffix(".tif")
    #                 output_path = r"C:\Users\emwil\Downloads\tes_cleaned_0_text" + os.sep + name + ".txt"
    #                 file = open(output_path, "w", encoding='utf-8-sig')
    #                 file.write(text)
    #                 file.close()
    #             except:
    #                 continue
    #
    # print("cleaned 1 ")
    # path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\1"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print(entry.name)
    #             try:
    #                 im = Image.open(entry.path)
    #                 text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    #                 name = entry.name.removesuffix(".tif")
    #                 output_path = r"C:\Users\emwil\Downloads\tes_cleaned_1_text" + os.sep + name + ".txt"
    #                 file = open(output_path, "w", encoding='utf-8-sig')
    #                 file.write(text)
    #                 file.close()
    #             except:
    #                 continue
    #
    #
    # print("cleaned 2 ")
    # path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\2"
    # suffix = ".tif"
    # output_path = r"C:\Users\emwil\Downloads\tes_cleaned_2_text"
    # get_text(path, output_path, suffix)
    # # with os.scandir(path) as it:
    # #     for entry in it:
    # #         if entry.name.endswith(".tif") and entry.is_file():
    # #             print(entry.name)
    # #             try:
    # #                 im = Image.open(entry.path)
    # #                 text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    # #                 name = entry.name.removesuffix(".tif")
    # #                 output_path = r"C:\Users\emwil\Downloads\tes_cleaned_2_text" + os.sep + name + ".txt"
    # #                 file = open(output_path, "w", encoding='utf-8-sig')
    # #                 file.write(text)
    # #                 file.close()
    # #             except:
    # #                 continue
    #
    #
    # print("cleaned 3 ")
    # path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\0"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print(entry.name)
    #             try:
    #                 im = Image.open(entry.path)
    #                 text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    #                 name = entry.name.removesuffix(".tif")
    #                 output_path = r"C:\Users\emwil\Downloads\tes_cleaned_3_text" + os.sep + name + ".txt"
    #                 file = open(output_path, "w", encoding='utf-8-sig')
    #                 file.write(text)
    #                 file.close()
    #             except:
    #                 continue
    #
    # print("cleaned 4 ")
    # path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\4"
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print(entry.name)
    #             try:
    #                 im = Image.open(entry.path)
    #                 text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    #                 name = entry.name.removesuffix(".tif")
    #                 output_path = r"C:\Users\emwil\Downloads\tes_cleaned_4_text" + os.sep + name + ".txt"
    #                 file = open(output_path, "w", encoding='utf-8-sig')
    #                 file.write(text)
    #                 file.close()
    #             except:
    #                 continue

    print("sharpened ")
    path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\sharpened"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".tif") and entry.is_file():
                # print(entry.name)
                try:
                    # im = Image.open(entry.path)
                    text = pytesseract.image_to_string(entry.path, lang="heb_tess_dict")
                    name = entry.name.removesuffix(".tif")
                    output_path = r"C:\Users\emwil\Downloads\tes_sharpened_text" + os.sep + name + ".txt"
                    file = open(output_path, "w", encoding='utf-8-sig')
                    file.write(text)
                    file.close()
                except:
                    continue


    print("contrasted ")
    path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\contrasted"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".tif") and entry.is_file():
                # print(entry.name)
                try:
                    # im = Image.open(entry.path)
                    text = pytesseract.image_to_string(entry.path, lang="heb_tess_dict")
                    name = entry.name.removesuffix(".tif")
                    output_path = r"C:\Users\emwil\Downloads\tes_contrasted_text" + os.sep + name + ".txt"
                    file = open(output_path, "w", encoding='utf-8-sig')
                    file.write(text)
                    file.close()
                except:
                    continue

    print("sharpened and contrasted")
    path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\sharp&contrast"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".tif") and entry.is_file():
                # print(entry.name)
                try:
                    # im = Image.open(entry.path)
                    text = pytesseract.image_to_string(entry.path, lang="heb_tess_dict")
                    name = entry.name.removesuffix(".tif")
                    output_path = r"C:\Users\emwil\Downloads\tes_sharp&contrast_text" + os.sep + name + ".txt"
                    file = open(output_path, "w", encoding='utf-8-sig')
                    file.write(text)
                    file.close()
                except:
                    continue

    # im = Image.open(r"C:\Users\emwil\Downloads\cleaned_achiezer-007.tif")
    # text = pytesseract.image_to_string(im, lang="heb_tess_dict")
    # # name = entry.name.removesuffix(".tif")
    # output_path = r"C:\Users\emwil\Downloads\inverted_achiezer-007.txt"
    # file = open(output_path, "w", encoding='utf-8-sig')
    # file.write(text)
    # file.close()
    # im_2 = Image.open(r"C:\Users\emwil\Downloads\אוצר החכמה_609879.jpg")
    # text = pytesseract.image_to_string(im, lang='heb')
    # print(text)
    # print(text_2)

    # print(pytesseract.image_to_boxes(im, lang ='heb'))

    # Get verbose data including boxes, confidences, line and page numbers
    # print(pytesseract.image_to_data(im, lang ='heb'))

    # print(pytesseract.image_to_data(im_2, lang = "heb_tess_dict"))

# 5	1	4	1	1	1	1309	598	197	41	91.613609	והשיבו:
# 5	1	4	2	1	7	31	    664	156	34	82.540298   והשיבו:

# Get information about orientation and script detection
# print(pytesseract.image_to_osd(im, lang='heb'))

def get_text(p, output, suffix):
    path = p
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(suffix) and entry.is_file():
                print(entry.name)
                try:
                    # im = Image.open(entry.path)
                    text = pytesseract.image_to_string(entry.path, lang="heb_tess_dict")
                    name = entry.name.removesuffix(suffix)
                    output_path = output + os.sep + name + ".txt"
                    file = open(output_path, "w", encoding='utf-8-sig')
                    file.write(text)
                    file.close()
                except:
                    continue


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def OldCoordinatesLetters(xml, png, output):
    xml = ET.parse(xml)
    root = xml.getroot()
    all_info_words = []
    words = []
    num_words = 0
    total_size = 0
    curr = []
    word = []
    total_letter_size = 0
    letter_count = 0
    letters = []
    letters_text_size_dim = []
    let_sizes = []
    obj = {}
    for i in range(1, 20000):
        obj['size_' + str(i)] = []
    for page in root:
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            if (num_words != 0 and len(curr) != 0):
                                # print(" newline ")
                                # new line is also the end of the word
                                # print("endword")
                                # get coordinate bounds of the word
                                left = 1000000
                                top = 1000000
                                right = 0
                                bottom = 0
                                for l in curr:
                                    if int(l.attrib['l']) < left:
                                        left = int(l.attrib['l'])
                                    if int(l.attrib['t']) < top:
                                        top = int(l.attrib['t'])
                                    if int(l.attrib['r']) > right:
                                        right = int(l.attrib['r'])
                                    if int(l.attrib['b']) > bottom:
                                        bottom = int(l.attrib['b'])
                                x0_y0_x1_y1 = [left, top, right, bottom]
                                width = right - left
                                height = bottom - top
                                word_size = width * height
                                # print(x0_y0_x1_y1)
                                # print(word_size)
                                total_size += word_size
                                # print("total_size: " + str(total_size))
                                words.append(curr)
                                s = "".join(word)
                                # print(s)
                                all_info_words.append([x0_y0_x1_y1, word_size, s])
                                num_words += 1
                                curr = []
                                word = []
                            for lang in line:
                                for char in lang:
                                    # print(char.items(), char.keys(), char)
                                    if char.text == ' ':
                                        # print("endword")
                                        # get coordinate bounds of the word
                                        left = 1000000
                                        top = 1000000
                                        right = 0
                                        bottom = 0
                                        for l in curr:
                                            if int(l.attrib['l']) < left:
                                                left = int(l.attrib['l'])
                                            if int(l.attrib['t']) < top:
                                                top = int(l.attrib['t'])
                                            if int(l.attrib['r']) > right:
                                                right = int(l.attrib['r'])
                                            if int(l.attrib['b']) > bottom:
                                                bottom = int(l.attrib['b'])
                                        x0_y0_x1_y1 = [left, top, right, bottom]
                                        width = right - left
                                        height = bottom - top
                                        word_size = width * height
                                        # print(x0_y0_x1_y1)
                                        # print(word_size)
                                        total_size += word_size
                                        # print("total_size: " + str(total_size))
                                        words.append(curr)
                                        s = "".join(word)
                                        # print(s)
                                        all_info_words.append([x0_y0_x1_y1, word_size, s])
                                        num_words += 1
                                        curr = []
                                        word = []
                                    else:
                                        if (char.text == '.' or char.text == ':'):
                                            continue
                                        curr.append(char)
                                        word.append(char.text)
                                        # print(char.text)
                                        letter_count += 1
                                        letters.append(char)
                                        left = int(char.attrib['l'])
                                        top = int(char.attrib['t'])
                                        right = int(char.attrib['r'])
                                        bottom = int(char.attrib['b'])
                                        x0_y0_x1_y1 = [left, top, right, bottom]
                                        w = right - left
                                        h = bottom - top
                                        let_size = w * h
                                        let_num = ord(char.text)
                                        # print(ord(char.text), char.text)
                                        obj['size_' + str(let_num)].append(let_size)
                                        let_sizes.append(let_size)
                                        letters_text_size_dim.append([char.text, let_size, x0_y0_x1_y1])
                                        total_letter_size += let_size

    avg_size = total_size / num_words
    # print(total_size, num_words)
    # print(avg_size)
    count = 0
    tot = 0
    img = Image.open(png).convert('RGBA')
    img_2 = img.copy()
    draw = ImageDraw.Draw(img_2)
    for i in all_info_words:
        # print(i)
        # print(i[0])
        # draw.rectangle(i[0], outline="red", width=2)
        tot += i[1]
        count += 1
    # print(tot/count)
    # img_2.show()
    # img = Image.open(r"C:\Users\emwil\Downloads\rashbameyuchas-030.png").convert('RGBA')

    print("     Now Letters:       ")
    print(total_letter_size / letter_count)
    mean = statistics.mean(let_sizes)
    std_dev = statistics.stdev(let_sizes)
    print("total mean: " + str(mean))
    print("total standard deviation: " + str(std_dev))
    means = {}
    std_devs = {}
    for i in range(1, 20000):
        if (len(obj['size_' + str(i)]) > 1):
            means['mean' + str(i)] = statistics.mean(obj['size_' + str(i)])
            std_devs['stdev' + str(i)] = statistics.stdev(obj['size_' + str(i)])
        else:
            means['mean' + str(i)] = 0
            std_devs['stdev' + str(i)] = 0
    for i in means:
        if means[i] != 0:
            print(means[i])
    # print(means)
    # print(std_devs)
    for i in letters_text_size_dim:
        # print("text: " + str(i[0]) + " size: " + str(i[1]))
        # print("dim: " + str((i[2])))
        num = ord(i[0])
        if (means['mean' + str(num)] == 0 or std_devs['stdev' + str(num)] == 0):
            continue
        # print(i[0])
        if (i[1] < (means['mean' + str(num)] + (3 * std_devs['stdev' + str(num)]))):
            draw.rectangle(i[2], outline="black", width=1)
        else:
            draw.rectangle(i[2], outline="red", width=1)
            # print("text: " + str(i[0]) + " size: " + str(i[1]))

    img_2.show()
    img_2.save(output, "PNG")
    # new = png.split("\\")
    # name = ""
    # for i in new:
    #     if (i.endswith(".png")):
    #         name = i.removesuffix(".png")
    #     if (i.endswith(".tif")):
    #         name = i.removesuffix(".tif")
    # full_name = r"C:\Users\emwil\Downloads\bolded letter average" + os.sep + name + "_test_bolded.png"
    # img_2.save(full_name, "PNG")

    # with open(img_2, 'rb') as f:
    #     data = f.read()
    #
    # with open('picture_out.png', 'wb') as f:
    #     f.write(data)

def bolded_by_height(xml, pic, output, rel_l, rel_r, rel_t, rel_b):
    xml = ET.parse(xml)
    root = xml.getroot()
    print(pic)
    height_cors = []
    total_width = 0
    total_height = 0
    # c_p = 0
    for page in root:
        total_width = int(page.attrib['width'])
        total_height = int(page.attrib['height'])
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            for lang in line:
                                for char in lang:
                                    # tot_text = tot_text + char.text
                                    if (char.text == '.' or char.text == ':' or char.text == ' '):
                                        # if (char.text == ':'):
                                        #     c_p = 0
                                        # else:
                                        #     c_p += 1
                                        continue
                                    # c_p += 1
                                    left = int(char.attrib['l'])
                                    top = int(char.attrib['t'])
                                    right = int(char.attrib['r'])
                                    bottom = int(char.attrib['b'])
                                    if ((char.text == '־' and left == 418 and top == 1238) or (
                                            char.text == 'י' and left == 415 and top == 1237) or (
                                            char.text == '~' and left == 608 and top == 215)):
                                        # print("skipping")
                                        continue
                                    x0_y0_x1_y1 = [left, top, right, bottom]
                                    w = right - left
                                    h = bottom - top
                                    # rel_l.append(float(left/total_width) * 100)
                                    # rel_r.append(float(right/total_width) * 100)
                                    # rel_t.append(float(top/total_height)* 100)
                                    # rel_b.append(float(bottom/total_height) * 100)
                                    rel_h = float(h/total_height) * 1000
                                    rel_w = float(w/total_width) * 1000
                                    # rel_height.append(rel_h)
                                    # rel_width.append(rel_w)
                                    # col_prox.append(c_p)
                                    height_cors.append([rel_h, x0_y0_x1_y1])
                                    # let_size = w * h
    # return rel_l, rel_r, rel_t, rel_b
    # return rel_height, rel_width, col_prox
    # for i in height_cors:
    #     print(i)
    # draw boxes around bolded words
    print("making pic for: " + output)
    img = Image.open(pic).convert('RGBA')
    img_2 = img.copy()
    draw = ImageDraw.Draw(img_2)
    for i in height_cors:
        if i[0] <= 10:
            draw.rectangle(i[1], outline="black", width=1)
        elif i[0] < 55 and i[0] > 40:
            draw.rectangle(i[1], outline="green", width=1)
        elif i[0] > 55:
            draw.rectangle(i[1], outline="red", width=1)
    img_2.show()
    img_2.save(output, "PNG")

# use this method to add features to the dataframe by passing in the xml and all the features you want to add as lists
# return the feature lists of data and add them to the dataframe
def add_feature(xml_path, line_width, line_height, chars_per_line, line_num):
    xml = ET.parse(xml_path)
    root = xml.getroot()
    print(xml_path)
    total_width = 0
    total_height = 0
    first_line = True
    line_chars = 0
    num_line = 0
    for page in root:
        total_width = int(page.attrib['width'])
        total_height = int(page.attrib['height'])
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            line_left = int(line.attrib['l'])
                            line_top = int(line.attrib['t'])
                            line_right = int(line.attrib['r'])
                            line_bottom = int(line.attrib['b'])
                            wid = line_right - line_left
                            high = line_bottom - line_top
                            rel_line_wid = float(wid/total_width) * 1000
                            rel__line_high = float(high/total_height) * 1000
                            if (first_line == False):
                                for i in range(line_chars):
                                    chars_per_line.append(line_chars)
                                num_line += 1
                            line_chars = 0
                            for lang in line:
                                for char in lang:
                                    first_line = False
                                    if (char.text == '.' or char.text == ':' or char.text == ' '):
                                        continue
                                    left = int(char.attrib['l'])
                                    top = int(char.attrib['t'])
                                    if ((char.text == '־' and left == 418 and top == 1238) or (
                                            char.text == 'י' and left == 415 and top == 1237) or (
                                            char.text == '~' and left == 608 and top == 215)):
                                        print("skipping")
                                        continue
                                    line_width.append(rel_line_wid)
                                    line_height.append(rel__line_high)
                                    line_chars += 1
                                    line_num.append(num_line)
    # add in chars per line from the last line of the page
    for i in range(line_chars):
        chars_per_line.append(line_chars)
    return line_width, line_height, chars_per_line, line_num


def bolded_by_width(xml, pic, output):
    xml = ET.parse(xml)
    root = xml.getroot()
    height_cors = []
    for page in root:
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            for lang in line:
                                for char in lang:
                                    # tot_text = tot_text + char.text
                                    if (char.text == '.' or char.text == ':' or char.text == ' '):
                                        continue
                                    left = int(char.attrib['l'])
                                    top = int(char.attrib['t'])
                                    right = int(char.attrib['r'])
                                    bottom = int(char.attrib['b'])
                                    # if ((char.text == '־' and left == 418 and top == 1238) or (
                                    #         char.text == 'י' and left == 415 and top == 1237) or (
                                    #         char.text == '~' and left == 608 and top == 215)):
                                    #     print("skipping")
                                    #     continue
                                    x0_y0_x1_y1 = [left, top, right, bottom]
                                    w = right - left
                                    # h = bottom - top
                                    height_cors.append([w, x0_y0_x1_y1])
                                    # let_size = w * h

    # draw boxes around bolded words
    print("making pic for: " + output)
    img = Image.open(pic).convert('RGBA')
    img_2 = img.copy()
    draw = ImageDraw.Draw(img_2)
    for i in height_cors:
        if i[0] < 25:
            draw.rectangle(i[1], outline="black", width=1)
        elif i[0] < 42 and i[0] > 25:
            draw.rectangle(i[1], outline="green", width=1)
        elif i[0] > 42:
            draw.rectangle(i[1], outline="red", width=1)
    # img_2.show()
    img_2.save(output, "PNG")




def CoordinatesUsingWords(xml, png):
    xml = ET.parse(xml)
    root = xml.getroot()
    all_info_words = []
    words = []
    num_words = 0
    total_size = 0
    curr = []
    word = []
    for page in root:
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            if (num_words != 0 and len(curr) != 0):
                                # print(" newline ")
                                # new line is also the end of the word
                                # print("endword")
                                # get coordinate bounds of the word
                                left = 1000000
                                top = 1000000
                                right = 0
                                bottom = 0
                                for l in curr:
                                    if int(l.attrib['l']) < left:
                                        left = int(l.attrib['l'])
                                    if int(l.attrib['t']) < top:
                                        top = int(l.attrib['t'])
                                    if int(l.attrib['r']) > right:
                                        right = int(l.attrib['r'])
                                    if int(l.attrib['b']) > bottom:
                                        bottom = int(l.attrib['b'])
                                x0_y0_x1_y1 = [left, top, right, bottom]
                                width = right - left
                                height = bottom - top
                                word_size = width * height
                                # print(x0_y0_x1_y1)
                                # print(word_size)
                                total_size += word_size
                                # print("total_size: " + str(total_size))
                                words.append(curr)
                                s = "".join(word)
                                # print(s)
                                all_info_words.append([x0_y0_x1_y1, word_size, s])
                                num_words += 1
                                curr = []
                                word = []
                            for lang in line:
                                for char in lang:
                                    # print(char.items(), char.keys(), char)
                                    if char.text == ' ':
                                        # print("endword")
                                        # get coordinate bounds of the word
                                        left = 1000000
                                        top = 1000000
                                        right = 0
                                        bottom = 0
                                        for l in curr:
                                            if int(l.attrib['l']) < left:
                                                left = int(l.attrib['l'])
                                            if int(l.attrib['t']) < top:
                                                top = int(l.attrib['t'])
                                            if int(l.attrib['r']) > right:
                                                right = int(l.attrib['r'])
                                            if int(l.attrib['b']) > bottom:
                                                bottom = int(l.attrib['b'])
                                        x0_y0_x1_y1 = [left, top, right, bottom]
                                        width = right - left
                                        height = bottom - top
                                        word_size = width * height
                                        # print(x0_y0_x1_y1)
                                        # print(word_size)
                                        total_size += word_size
                                        # print("total_size: " + str(total_size))
                                        words.append(curr)
                                        s = "".join(word)
                                        # print(s)
                                        all_info_words.append([x0_y0_x1_y1, word_size, s])
                                        num_words += 1
                                        curr = []
                                        word = []
                                    else:
                                        if (char.text == '.' or char.text == ':'):
                                            continue
                                        curr.append(char)
                                        word.append(char.text)
                                        # print(char.text)

    avg_size = total_size / num_words
    # print(total_size, num_words)
    # print(avg_size)
    img = Image.open(png).convert('RGBA')
    img_2 = img.copy()
    draw = ImageDraw.Draw(img_2)
    for i in all_info_words:
        if (i[1] > 2 * avg_size):
            draw.rectangle(i[0], outline="black", width=2)
        else:
            draw.rectangle(i[0], outline="red", width=2)
    img_2.show()
    # img_2.save(output, "PNG")

    # new = png.split("\\")
    # name = ""
    # for i in new:
    #     if (i.endswith(".png")):
    #         name = i.removesuffix(".png")
    #     if (i.endswith(".tif")):
    #         name = i.removesuffix(".tif")
    # full_name = r"C:\Users\emwil\Downloads\bolded letter average" + os.sep + name + "_test_bolded.png"
    # img_2.save(full_name, "PNG")

def getCoordinates(xml, png, output, n, chars, counter, count):
    text = " "
    xml = ET.parse(xml)
    root = xml.getroot()
    all_info_words = []
    words = []
    num_words = 0
    total_size = 0
    curr = []
    word = []
    total_letter_size = 0
    letter_count = 0
    letters = []
    text_size_dim = []
    let_sizes = []
    obj = {}
    lc = 0

    # seperate name and page number
    name = n.split('-')[0]
    page_num = n.split('-')[1]
    print("name and page: " + str(name) + " " + str(page_num))

    for i in range(1, 20000):
        obj['size_' + str(i)] = []
    total_width = 0
    total_height = 0
    for page in root:
        total_width = int(page.attrib['width'])
        total_height = int(page.attrib['height'])
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            # print(type(os.linesep), type('\n'))
                            # tot_text = tot_text + '\n'
                            lc = 0
                            text = text + '\n'
                            for lang in line:
                                for char in lang:
                                    count += 1
                                    # tot_text = tot_text + char.text
                                    text = text + char.text
                                    if (char.text == '.' or char.text == ':' or char.text == ' '):
                                        lc += 1
                                        continue
                                    left = int(char.attrib['l'])
                                    top = int(char.attrib['t'])
                                    right = int(char.attrib['r'])
                                    bottom = int(char.attrib['b'])
                                    # if ((char.text == '־' and left == 418 and top == 1238) or (char.text == 'י' and left == 415 and top == 1237) or (char.text == '~' and left == 608 and top == 215)):
                                    #     print("skipping")
                                    #     continue
                                    letter_count += 1
                                    letters.append(char)
                                    x0_y0_x1_y1 = (left, top, right, bottom)
                                    w = right - left
                                    h = bottom - top
                                    let_size = w * h
                                    let_num = ord(char.text)
                                    # print(ord(char.text), char.text)
                                    obj['size_' + str(let_num)].append(let_size)
                                    let_sizes.append(let_size)
                                    text_size_dim.append([char.text, let_size, x0_y0_x1_y1])
                                    total_letter_size += let_size
                                    # rel_h = float(h/total_height) * 1000
                                    # rel_w = float(w/total_width) * 1000
                                    # total_num_chars = len(c) + 1
                                    # Order is: 1) Book Name, 2) Page Number, 3) Char, 4) Bolded, 5) Coordinates, 6) Size, 7) Height, 8) Width
                                    # 9) ASCII val, 10) counter, 11) Line Location, 12) Left, 13) Right, 14) Top, 15) Bottom
                                    chars.append([name, page_num, char.text, False, x0_y0_x1_y1, let_size, h, w, let_num, counter, lc, left, right, top, bottom])
                                    # chars.append([char.text, counter, let_size, x0_y0_x1_y1, h, w, let_num, name, page_num, False])
                                    counter += 1
                                    # line_loc.append(lc)
                                    lc += 1

                                    # to check using height instead of total size
                                    # let_height = h
                                    # obj['size_' + str(let_num)].append(let_height)
                                    # let_sizes.append(let_height)
                                    # text_size_dim.append([char.text, let_height, x0_y0_x1_y1])
                                    # total_letter_size += let_height

    # print("chars so far: " + str(counter))
    # print("total num of chars so far: " + str(len(chars)))
    # print("printing text: ")
    # print(text)

    # create text files from abbyy xml
    # folder = r"C:\Users\emwil\Downloads\abbyy_text"
    # file_path = folder + os.sep + n + ".txt"
    # file = open(file_path, "w+", encoding='utf-8-sig')
    # file.write(text)
    # file.close()

    # draw boxes around bolded words
    if (letter_count == 0):
        print("Did not recongnize ANY letters from this document")
    else:
        # print("total text so far: " + str(len(tot_text)))

        # img = Image.open(png).convert('RGBA')
        # img_2 = img.copy()
        # draw = ImageDraw.Draw(img_2)
        # mean = statistics.mean(let_sizes)
        # std_dev = statistics.stdev(let_sizes)

        # print("total mean: " + str(mean))
        # print("total standard deviation: " + str(std_dev))

        # means = {}
        # std_devs = {}
        # for i in range(1, 20000):
        #     if (len(obj['size_' + str(i)]) > 1):
        #         means['mean' + str(i)] = statistics.mean(obj['size_' + str(i)])
        #         std_devs['stdev' + str(i)] = statistics.stdev(obj['size_' + str(i)])
        #     else:
        #         means['mean' + str(i)] = 0
        #         std_devs['stdev' + str(i)] = 0

        # print("number of letters are: " + str(len(letters)))
        # print(" ")
        # for i in means:
        #     if means[i] != 0:
        #         print(means[i])
        # print(" now std devs:    ")
        # for i in std_devs:
        #     if (std_devs[i] != 0):
        #         print(std_devs[i])
        # print(" ")
        # for i in text_size_dim:
        #     num = ord(i[0])
        #     if (means['mean' + str(num)] == 0 or std_devs['stdev'+str(num)] == 0):
        #         continue
        #     if (i[1] < (means['mean'+str(num)]+(3* std_devs['stdev'+str(num)]))):
        #
        #         draw.rectangle(i[2], outline="black", width=1)
        #     else:
        #         # print("letter: " + i[0])
        #         # print("actual size: " + str(i[1]))
        #         # print("mean + 3 std devs: " + str((means['mean'+str(num)]+(3* std_devs['stdev'+str(num)]))))
        #         draw.rectangle(i[2], outline="red", width=1)
        #         # print("text: " + str(i[0]) + " size: " + str(i[1]))
        #
        # # img_2.show()
        # img_2.save(output, "PNG")
        return count
        # num of charachters
        # print("num of charachters in this doc: " + str(len(text_size_dim)) + " in " + n)
        # print("total num of charachters seen so far: " + str(len(c)))
        # return text




        # new = png.split("\\")
        # name = ""
        # for i in new:
        #     if (i.endswith(".png")):
        #         name = i.removesuffix(".png")
        #     if (i.endswith(".tif")):
        #         name = i.removesuffix(".tif")
        # full_name = r"C:\Users\emwil\Downloads\bolded letter average" + os.sep + name + "_test_bolded.png"
        # img_2.save(full_name, "PNG")

        # with open(img_2, 'rb') as f:
        #     data = f.read()
        #
        # with open('picture_out.png', 'wb') as f:
        #     f.write(data)

def move_files():
    p = r"C:\Users\emwil\Downloads\processed_images\cleaned_images"
    with os.scandir(p) as it:
        for entry in it:
            if entry.name.endswith(".png") and entry.is_file():
                image = cv2.imread(entry.path)
                new_out_path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images_png" + os.sep + entry.name
                cv2.imwrite(new_out_path, image)


def clean_image(input, output, name):
    image = cv2.imread(input, 0)
    # num_comps, labeled_pixels, comp_stats, comp_centroids = \
    #     cv2.connectedComponentsWithStats(image, connectivity=4)
    # min_comp_area = 10  # pixels
    # # get the indices/labels of the remaining components based on the area stat
    # # (skip the background component at index 0)
    # remaining_comp_labels = [i for i in range(1, num_comps) if comp_stats[i][4] >= min_comp_area]
    # # filter the labeled pixels based on the remaining labels,
    # # assign pixel intensity to 255 (uint8) for the remaining pixels
    # sharpen = np.where(np.isin(labeled_pixels, remaining_comp_labels) == True, 255, 0).astype('uint8')

    # cv2.imshow('original', image)
    # cv2.waitKey(10)

    # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    # out_pa = output + os.sep + name
    # cv2.imwrite(out_pa, sharpen)


    # cv2.imshow('sharpened', sharpen)

    # cv2.waitKey(20)


    img = Image.open(input)

    # ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    # titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    # images = [img, thresh1, thresh2, thresh3, thresh4]
    # for i in range(5):
    #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    #     output_p = output + os.sep + str(i) + os.sep + name
    #     print(output_p)
    #     # plt.imsave(output_p)
    #     cv2.imwrite(output_p, images[i])
    # plt.show()
    # cv2.waitKey()

    enhancer = ImageEnhance.Sharpness(img)
    a = enhancer.enhance(4)
    output_pa = output + os.sep + "sharpened" + os.sep + name
    a.save(output_pa)

    #also try out contrast
    contraster = ImageEnhance.Contrast(img)
    b = contraster.enhance(4)
    output_pat = output + os.sep + "contrasted" + os.sep + name
    b.save(output_pat)

    # now try both
    sharper = ImageEnhance.Sharpness(img)
    c = sharper.enhance(4)
    contrast = ImageEnhance.Contrast(c)
    c = contrast.enhance(4)
    out_path = output + os.sep + "sharp&contrast" + os.sep + name
    c.save(out_path)


    # a.show('enhanced')
    # output_p = output + os.sep +
    # a.save(output)
    # Image._show(img)

    # img = cv2.imread(r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-025.png", 1)
    # img = cv2.resize(img, (0,0), fx=.2, fy=.2)
    # cv2.imshow('Ein Yitzchak', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    # image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for c in contours:
    #     # get the bounding rect
    #     x, y, w, h = cv2.boundingRect(c)
    #     # draw a white rectangle to visualize the bounding rect
    #     cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)
    # cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    # cv2.imwrite(r"C:\Users\emwil\Downloads\random images to send\einyitzchak-025.png", img)

def gc(xml, png, output):
    text = " "
    xml = ET.parse(xml)
    root = xml.getroot()
    all_info_words = []
    words = []
    num_words = 0
    total_size = 0
    curr = []
    word = []
    total_letter_size = 0
    letter_count = 0
    letters = []
    text_size_dim = []
    let_sizes = []
    obj = {}
    lc = 0


    # for i in range(1, 20000):
    #     obj['size_' + str(i)] = []
    for page in root:
        for block in page:
            for t in block:
                if "text" in t.tag:
                    for par in t:
                        for line in par:
                            # print(type(os.linesep), type('\n'))
                            # tot_text = tot_text + '\n'
                            lc = 0
                            text = text + '\n'
                            for lang in line:
                                for char in lang:
                                    # tot_text = tot_text + char.text
                                    text = text + char.text
                                    if (char.text == '.' or char.text == ':' or char.text == ' '):
                                        lc += 1
                                        continue
                                    left = int(char.attrib['l'])
                                    top = int(char.attrib['t'])
                                    right = int(char.attrib['r'])
                                    bottom = int(char.attrib['b'])
                                    # if ((char.text == '־' and left == 418 and top == 1238) or (
                                    #         char.text == 'י' and left == 415 and top == 1237) or (
                                    #         char.text == '~' and left == 608 and top == 215)):
                                    #     print("skipping")
                                    #     continue
                                    letter_count += 1
                                    letters.append(char)
                                    x0_y0_x1_y1 = (left, top, right, bottom)
                                    w = right - left
                                    h = bottom - top
                                    let_size = w * h
                                    let_num = ord(char.text)
                                    # print(ord(char.text), char.text)
                                    # obj['size_' + str(let_num)].append(let_size)
                                    let_sizes.append(let_size)
                                    text_size_dim.append([char.text, let_size, x0_y0_x1_y1])
                                    total_letter_size += let_size
                                    # total_num_chars = len(c) + 1
                                    # Order is: 1) Book Name, 2) Page Number, 3) Char, 4) Bolded, 5) Coordinates, 6) Size, 7) Height, 8) Width
                                    # 9) ASCII val, 10) counter, 11) Line Location, 12) Left, 13) Right, 14) Top, 15) Bottom
                                    # chars.append(
                                    #     [name, page_num, char.text, False, x0_y0_x1_y1, let_size, h, w, let_num,
                                    #      counter, lc, left, right, top, bottom])
                                    # # chars.append([char.text, counter, let_size, x0_y0_x1_y1, h, w, let_num, name, page_num, False])
                                    # counter += 1
                                    # line_loc.append(lc)
                                    lc += 1

                                    # to check using height instead of total size
                                    # let_height = h
                                    # obj['size_' + str(let_num)].append(let_height)
                                    # let_sizes.append(let_height)
                                    # text_size_dim.append([char.text, let_height, x0_y0_x1_y1])
                                    # total_letter_size += let_height

    # print("chars so far: " + str(counter))
    # print("total num of chars so far: " + str(len(chars)))
    # print("printing text: ")
    # print(text)

    # create text files from abbyy xml
    # folder = r"C:\Users\emwil\Downloads\abbyy_text"
    # file_path = folder + os.sep + n + ".txt"
    # file = open(file_path, "w+", encoding='utf-8-sig')
    # file.write(text)
    # file.close()

    # draw boxes around bolded words
    if (letter_count == 0):
        print("Did not recongnize ANY letters from this document")
    else:

        img = Image.open(png).convert('RGBA')
        img_2 = img.copy()
        draw = ImageDraw.Draw(img_2)
        mean = statistics.mean(let_sizes)
        std_dev = statistics.stdev(let_sizes)

    # print("total mean: " + str(mean))
    # print("total standard deviation: " + str(std_dev))

        means = {}
        std_devs = {}
        for i in range(1, 20000):
            if (len(obj['size_' + str(i)]) > 1):
                means['mean' + str(i)] = statistics.mean(obj['size_' + str(i)])
                std_devs['stdev' + str(i)] = statistics.stdev(obj['size_' + str(i)])
            else:
                means['mean' + str(i)] = 0
                std_devs['stdev' + str(i)] = 0

        print("number of letters are: " + str(len(letters)))
        print(" ")
        for i in means:
            if means[i] != 0:
                print(means[i])
        print(" now std devs:    ")
        for i in std_devs:
            if (std_devs[i] != 0):
                print(std_devs[i])
        print(" ")
        for i in text_size_dim:
            num = ord(i[0])
            if (means['mean' + str(num)] == 0 or std_devs['stdev'+str(num)] == 0):
                continue
            if (i[1] < (means['mean'+str(num)]+(3* std_devs['stdev'+str(num)]))):

                draw.rectangle(i[2], outline="black", width=1)
            else:
                # print("letter: " + i[0])
                # print("actual size: " + str(i[1]))
                # print("mean + 3 std devs: " + str((means['mean'+str(num)]+(3* std_devs['stdev'+str(num)]))))
                draw.rectangle(i[2], outline="red", width=1)
        print("text: " + str(i[0]) + " size: " + str(i[1]))

        img_2.show()
        img_2.save(output, "PNG")


def fill_bold(csv_path):
    print("loading in prev csv file")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    list = [False] * 293030
    print(len(list))
    df['Bolded'] = list
    # df = pd.read_excel(csv_path)
    # pd.read_excel()

    # print("printing values:")
    # for i in range(1,9):
    #     print(i, df[7][i])

    # print(str(df['Bolded'][3922]))
    # print(str(df[7]['Book Name']))
    # print(str(df[7][9]))

    up_to_num = 0
    path = r"C:\Users\emwil\Downloads\abbyy_text"
    bool = True
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".txt") and entry.is_file():
                # if (entry.path != r"C:\Users\emwil\Downloads\abbyy_text\kerenora2-002.txt"):
                #     continue
                print("up to file: " + entry.name)
                # input("Press Enter when done adding stars...")
                file = open(entry.path, 'r', encoding='utf-8-sig')
                contents = file.read()
                # print("up to: " + str(up_to_num))
                # up_to_num = 117762
                for i in contents:
                    if (bool and up_to_num == 141709):
                        up_to_num -= 4
                        bool = False
                    if (i == '.' or i == ':' or i == ' ' or i == '\n'):
                        continue
                    if (i == '~'):
                        # up to (and not including) ohelmosheresponsa-018.txt
                        # mark previous index bold
                        # print("before " + str(df['Bolded'][up_to_num - 1]))
                        # print("up to: " + str(up_to_num-1))
                        # print(df['char'][up_to_num-1])
                        df['Bolded'][up_to_num - 1] = True
                        # print("found bold at index: " + str(up_to_num - 1))
                        # print("after " + str(df['Bolded'][up_to_num - 1]))
                    else:
                        up_to_num += 1
                # df.to_excel(csv_path)
                df.to_csv(csv_path, encoding='utf-8-sig')


def visualize(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    # print(df.axes)
    # df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis='columns', inplace=True)
    # print(df)
    # df[['Left', 'Top', 'Right', 'Bottom']] = df.cors.str.split(",", expand=True)
    # print(df.axes)
    # df['Left'] = df['Left'].str.replace('[', '')
    # df['Bottom'] = df['Bottom'].str.replace(']', '')

    # pd.set_option('display.max_columns', None)
    df.drop('Unnamed: 0', axis='columns', inplace=True)
    # print(df.head(3))
    # display(df)
    # df.to_csv(r'C:\Users\emwil\Downloads\csv_data.csv', encoding='utf-8-sig')
    # df.plot.hist(by='Bolded')
    # p = df['height'].plot.hist(bins=300)
    # df['height'].plot(x=['Bolded'], kind='hist', legend=True)
    # plt.hist(df, bins=2, x='height')

    # df['Bolded'] = df['Bolded'].replace(True, 1)
    # df['Bolded'] = df['Bolded'].replace(False, 0)
    # fig = plt.figure()
    # a = fig.add_axes([0,1])
    # a.bar(width= df['Bolded'], height =df['size'],)
    # b = df['Bolded']
    # p = df['width']

    # plt.bar(x= b, height=p)

    # scatter plot
    # plt.scatter(b, p)
    # plt.xticks([0,1])
    # plt.xlabel("Bolded")
    # plt.ylabel("Width")
    # plt.show()

    # plt.style.use('fivethirtyeight')
    plt.style.use('seaborn')
    color = '#fc4f30'


    #HIST
    bold = df['Bolded']
    width_bolded = df.loc[df['Bolded'] == True, 'width']
    width_unbolded = df.loc[df['Bolded'] == False, 'width']
    width = df['width']
    height = df['height']
    height_bolded = df.loc[df['Bolded'] == True, 'height']
    height_unbolded = df.loc[df['Bolded'] == False, 'height']
    size = df['size']
    size_bolded = df.loc[df['Bolded'] == True, 'size']
    size_unbolded = df.loc[df['Bolded'] == False, 'size']



    #width
    # fig, ax = plt.subplots(3,3)
    # ax[0,0].hist(width, bins=7, edgecolor='black')
    # ax[0,0].set_title("Width Plot")
    # ax[0,0].set_xlabel("Width")
    # ax[0,0].set_ylabel("Number of Charachters in Bin")
    # ax[0,0].tight_layout()
    # median_age = statistics.mean(width)
    # ax[0,0].set_axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.show()

    # plt.hist(width, bins=7, edgecolor='black')
    # plt.title("Width Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(width)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width.png", bbox_inches='tight')
    # plt.show()
    # plt.close()

    #
    # plt.hist(width, bins=7, edgecolor='black', log=True)
    # plt.title("Width Log Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(width)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # bins = [0, 5, 10, 15, 20, 25, 30]
    # plt.hist(width, bins=bins, edgecolor='black')
    # plt.title("Specific Width Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(width)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # # Now for bolded and unbolded
    # plt.hist(width_unbolded, bins=7, edgecolor='black')
    # plt.title("Width of Unbolded Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(width_unbolded)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_unbolded.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # plt.hist(width_unbolded, bins=7, edgecolor='black', log=True)
    # plt.title("Width of Unbolded Log Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(width_unbolded)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_unbolded_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # bins = [0, 5, 10, 15, 20, 25, 30]
    # plt.hist(width_unbolded, bins=bins, edgecolor='black')
    # plt.title("Specific Width Unbolded Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(width_unbolded)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width__unbolded_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # plt.hist(width_bolded, bins=7, edgecolor='black')
    # plt.title("Width of Bolded Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(width_bolded)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_bolded.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # plt.hist(width_bolded, bins=7, edgecolor='black', log=True)
    # plt.title("Width of Bolded Log Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(width_bolded)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_bolded_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # plt.hist(width_bolded, bins=bins, edgecolor='black')
    # plt.title("Specific Width Bolded Plot")
    # plt.xlabel("Width")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(width_bolded)
    # plt.axvline(median_age, color=color, label='Width Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\width_bolded_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # print("Width:")
    # print("Mean Reg: " + str(statistics.mean(width)))
    # print("Mean unbolded: " + str(statistics.mean(width_unbolded)))
    # print("Mean bolded: " + str(statistics.mean(width_bolded)))
    # print("Mode Reg: " + str(statistics.mode(width)))
    # print("Mode unbolded: " + str(statistics.mode(width_unbolded)))
    # print("Mode bolded: " + str(statistics.mode(width_bolded)))
    # print("Std dev Reg: " + str(statistics.stdev(width)))
    # print("Std dev unbolded: " + str(statistics.stdev(width_unbolded)))
    # print("Std dev bolded: " + str(statistics.stdev(width_bolded)))
    #
    # #height
    # plt.hist(height, bins=7, edgecolor='black')
    # plt.title("Height Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(height)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # plt.hist(height, bins=7, edgecolor='black', log=True)
    # plt.title("Height Log Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(height)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\Height_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # bins = [0, 5, 10, 15, 20, 25, 30,35, 40, 45, 50]
    # plt.hist(height, bins=bins, edgecolor='black')
    # plt.title("Specific Height Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(height)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # # Now for bolded and unbolded
    # plt.hist(height_unbolded, bins=7, edgecolor='black')
    # plt.title("Height of Unbolded Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(height_unbolded)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_unbolded.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # plt.hist(height_unbolded, bins=7, edgecolor='black', log=True)
    # plt.title("Height of Unbolded Log Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(height_unbolded)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_unbolded_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # bins = [0, 5, 10, 15, 20, 25, 30,35, 40, 45, 50]
    # plt.hist(height_unbolded, bins=bins, edgecolor='black')
    # plt.title("Specific Height Unbolded Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(height_unbolded)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_unbolded_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # plt.hist(height_bolded, bins=7, edgecolor='black')
    # plt.title("Height of Bolded Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(height_bolded)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_bolded.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # plt.hist(height_bolded, bins=7, edgecolor='black', log=True)
    # plt.title("Height of Bolded Log Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(height_bolded)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_bolded_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    # #
    # bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    # plt.hist(height_bolded, bins=bins, edgecolor='black')
    # plt.title("Specific Height Bolded Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(height_bolded)
    # plt.axvline(median_age, color=color, label='Height Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_bolded_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # print("Height:")
    # print("Mean Reg: " + str(statistics.mean(height)))
    # print("Mean unbolded: " + str(statistics.mean(height_unbolded)))
    # print("Mean bolded: " + str(statistics.mean(height_bolded)))
    # print("Mode Reg: " + str(statistics.mode(height)))
    # print("Mode unbolded: " + str(statistics.mode(height_unbolded)))
    # print("Mode bolded: " + str(statistics.mode(height_bolded)))
    # print("Std dev Reg: " + str(statistics.stdev(height)))
    # print("Std dev unbolded: " + str(statistics.stdev(height_unbolded)))
    # print("Std dev bolded: " + str(statistics.stdev(height_bolded)))
    #
    # # Size

    #combined
    fig, ax = plt.subplots(3,3)
    fig.suptitle("Size Plot")
    ax[0,0].hist(size, bins=7, edgecolor='black')
    ax[0,0].set_title("Reg")
    # ax[0,0].set_xlabel("Size")
    ax[0,0].set_ylabel("Total")
    ax[0,1].hist(size, bins=7, edgecolor='black', log=True)
    ax[0,1].set_title("Log")
    # ax[0,1].set_xlabel("Size Log")
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ax[0,2].hist(size, bins=bins, edgecolor='black')
    ax[0,2].set_title("Specific")
    # ax[0,2].set_xlabel("Size")

    ax[1,0].hist(size_unbolded, bins=7, edgecolor='black')
    ax[1,0].set_ylabel("Unbolded")
    ax[1,1].hist(size_unbolded, bins=7, edgecolor='black', log=True)
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ax[1,2].hist(size_unbolded, bins=bins, edgecolor='black')

    ax[2,0].hist(size_bolded, bins=7, edgecolor='black')
    ax[2,0].set_ylabel("Bolded")
    ax[2,1].hist(size_bolded, bins=7, edgecolor='black', log=True)
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ax[2,2].hist(size_bolded, bins=bins, edgecolor='black')
    plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_comp.png", bbox_inches='tight')
    plt.show()


    # plt.hist(size, bins=7, edgecolor='black')
    # plt.title("Size Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(size)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # plt.hist(size, bins=7, edgecolor='black', log=True)
    # plt.title("Size Log Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(size)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # plt.hist(size, bins=bins, edgecolor='black')
    # plt.title("Specific Size Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(size)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    #
    # # Now for bolded and unbolded
    # plt.hist(size_unbolded, bins=7, edgecolor='black')
    # plt.title("Size of Unbolded Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(size_unbolded)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_unbolded.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # #
    # plt.hist(size_unbolded, bins=7, edgecolor='black', log=True)
    # plt.title("Size of Unbolded Log Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(size_unbolded)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_unbolded_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # # bins = [0, 5, 10, 15, 20, 25, 30]
    # bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # plt.hist(size_unbolded, bins=bins, edgecolor='black')
    # plt.title("Specific Size Unbolded Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(size_unbolded)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_unbolded_spec.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    #
    # plt.hist(size_bolded, bins=7, edgecolor='black')
    # plt.title("Size of Bolded Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(size_bolded)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_bolded.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # #
    # plt.hist(size_bolded, bins=7, edgecolor='black', log=True)
    # plt.title("Size of Bolded Log Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # median_age = statistics.mean(size_bolded)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_bolded_log.png", bbox_inches='tight')
    # # plt.show()
    # plt.close()
    #
    # #
    # bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # plt.hist(size_bolded, bins=bins, edgecolor='black')
    # plt.title("Size Bolded Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # median_age = statistics.mean(size_bolded)
    # plt.axvline(median_age, color=color, label='Size Median', linewidth=2)
    # plt.legend()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_bolded_spec.png", bbox_inches='tight')
    # plt.close()
    # # plt.show()
    #
    # print("Size:")
    # print("Mean Reg: " + str(statistics.mean(size)))
    # print("Mean unbolded: " + str(statistics.mean(size_unbolded)))
    # print("Mean bolded: " + str(statistics.mean(size_bolded)))
    # print("Mode Reg: " + str(statistics.mode(size)))
    # print("Mode unbolded: " + str(statistics.mode(size_unbolded)))
    # print("Mode bolded: " + str(statistics.mode(size_bolded)))
    # print("Std dev Reg: " + str(statistics.stdev(size)))
    # print("Std dev unbolded: " + str(statistics.stdev(size_unbolded)))
    # print("Std dev bolded: " + str(statistics.stdev(size_bolded)))



    # plt.hist(height, bins=7, edgecolor='black')
    # plt.title("Height Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height.png", bbox_inches='tight')
    # plt.show()
    #
    # plt.hist(height, bins=7, edgecolor='black', log=True)
    # plt.title("Height Log Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_log.png", bbox_inches='tight')
    # plt.show()

    # bins = [0, 7, 14, 21, 28, 35, 42, 49, 56]
    # bins =[0,5,10,15,20,25,30,35]
    # plt.hist(width, bins=bins, edgecolor='black')
    # plt.title("Specific Height Plot")
    # plt.xlabel("Height")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\height_spec.png", bbox_inches='tight')
    # plt.show()

    # size
    # plt.hist(size, bins=7, edgecolor='black')
    # plt.title("Size Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size.png", bbox_inches='tight')
    # plt.show()
    #
    # plt.hist(size, bins=7, edgecolor='black', log=True)
    # plt.title("Size Log Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin (Log)")
    # plt.tight_layout()
    # # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_log.png", bbox_inches='tight')
    # plt.show()
    #
    # bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # # bins = [0, 200, 400, 600, 600, 800, 1000, 1200, 1400]
    # # bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    # plt.hist(size, bins=bins, edgecolor='black')
    # plt.title("Specific Size Plot")
    # plt.xlabel("Size")
    # plt.ylabel("Number of Charachters in Bin")
    # plt.tight_layout()
    # # plt.savefig(r"C:\Users\emwil\Downloads\Data Pics\size_spec.png", bbox_inches='tight')
    # plt.show()

    # plt.hist(data= df, x='Bolded', y='height')
    # plt.show()
    # plt.close()
    # plt.hist(df, x='height', bins=20)
    # plt.show()

    # df.drop('Unnamed: 0.1.1', axis='columns', inplace=True)
    # print(df)
    # df.drop('Unnamed: 0.1', axis='columns', inplace=True)
    # print(df)

def ground_truth_bold(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    prev_name = " "
    print(type(df['Coordinates'][3]))
    print(len(df['Bolded']))

    # first lets replace each cors with a tuple
    # tup_cor = []
    # for index, rows in df.iterrows():
    #     print(rows['cors'])
    #     cor = rows['cors'].removeprefix('[').removesuffix(']').split(', ')
    #     c = []
    #     for i in cor:
    #         c.append(int(i))
    #     tup_cor.append(tuple(c))
    # df['cors'] = tup_cor
    # df.to_csv(path, encoding='utf-8-sig')
    # print("done replacing column with tuples")
    count = 0
    # index, rows = df.iterrows()
    # print(type(index))
    # print(type(rows))
    pic_bold_cors = []
    # base = 10
    updated_cors = []
    b = 6
    for rows in df.itertuples():
        # print(rows)
        # print(rows[b])
        name = rows[b] + "-" + str(rows[b+1]).zfill(3) + ".tif"
        # print(name)
        # if (rows[1] == 141709):
        #     next(df.itertuples())
        #     next(df.itertuples())
        #     next(df.itertuples())
        #     next(df.itertuples())
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b+1]).zfill(2) + ".tif"
        if (rows[b] == "zikukindenura"):
            name = rows[b] + "-" + str(rows[b+1]) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        # print([pic, rows[17], rows[11]])
        # print(rows[b+4])
        # print(type(rows[b+4]))
        cor = rows[b+4].removeprefix('(').removesuffix(')').split(', ')
        # cor = rows[b-4].removeprefix('[').removesuffix(']').split(', ')
        c = []
        for i in cor:
            # c.__add__(int(i))
            c.append(int(i))
        pic_bold_cors.append([pic, rows[b+3], c])
        # updated_cors.append(c)
    print("done making list")
    # df['cors'] = updated_cors
    # df.to_csv(path, encoding='utf-8-sig')
    # print(type(df['cors'][3]))


    prev_pic = (pic_bold_cors[0])[0]
    output = prev_pic.replace("processed_images\processed_images", "Ground Truth Pics")
    print(output)
    img = Image.open(prev_pic).convert('RGBA')
    draw = ImageDraw.Draw(img)
    for a in pic_bold_cors:
        # print(a)
        if a[0] != prev_pic:
            output = prev_pic.replace("processed_images\processed_images", "Ground Truth Pics")
            img.save(output)
            img.close()
            print("Next Pic: " + a[0])
            img = Image.open(a[0]).convert('RGBA')
            prev_pic = a[0]
            draw = ImageDraw.Draw(img)
        # img = Image.open(a[0]).convert('RGBA')
        if (a[1] == True):
            draw.rectangle(a[2], outline='red', width=1)
        elif (a[1] == False):
            draw.rectangle(a[2], outline='black', width=1)
    output = prev_pic.replace("processed_images\processed_images", "Ground Truth Pics")
    # img.show()
    img.save(output)
    img.close()


        # if (name != prev_name):
        #     print(name)
        # prev_name = name
        # print(pic)
        # img_2 = img.copy()
        # cor = rows['cors'].removeprefix('[')
        # cor = cor.removesuffix(']')
        # l = cor.split(', ')
        # c = []
        # for i in l:
        #     c.append(int(i))
        # t = tuple(c)

        # output = r"C:\Users\emwil\Downloads\Ground Truth Pics" + os.sep + name.removesuffix(".tif") + ".png"
        # print(output)
        # print(count)
        # count += 1
        # img.save(pic)

def find_clumps(path):
    two_cols = pd.read_csv(path, encoding='utf-8-sig')
    pic_bold_cors_height = []
    # base = 10
    b = 2
    for rows in two_cols.itertuples():
        # print(rows)
        # print(rows[b])
        # print(rows[b])
        name = rows[b] + "-" + str(rows[b + 1]).zfill(3) + ".tif"
        # print(name)
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b + 1]).zfill(2) + ".tif"
        if (rows[b] == "zikukindenura"):
            name = rows[b] + "-" + str(rows[b + 1]) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        cor = [rows[13], rows[15], rows[14], rows[16]]
        pic_bold_cors_height.append([pic, rows[b + 3], cor, rows[b + 15]])
    print(pic_bold_cors_height[3])
    input("press enter:")

    prev_pic = (pic_bold_cors_height[0])[0]
    output = prev_pic.replace("processed_images\processed_images", "height pics")
    # print(output)
    # print(prev_pic)
    img = Image.open(prev_pic).convert('RGBA')
    draw = ImageDraw.Draw(img)
    # print("creted draw")
    for a in pic_bold_cors_height:
        # print(a)
        if a[0] != prev_pic:
            output = prev_pic.replace("processed_images\processed_images", "height pics")
            img.save(output)
            img.close()
            print("Next Pic: " + a[0])
            img = Image.open(a[0]).convert('RGBA')
            prev_pic = a[0]
            draw = ImageDraw.Draw(img)
        # img = Image.open(a[0]).convert('RGBA')
        if (a[1] == True):
            if (a[3] <= 10):
                draw.rectangle(a[2], outline='green', width=2)
            else:
                draw.rectangle(a[2], outline='red', width=2)
        elif (a[1] == False):
            draw.rectangle(a[2], outline='black', width=1)
    output = prev_pic.replace("processed_images\processed_images", "height pics")
    img.save(output)
    img.close()

if __name__ == '__main__':
    # do stuff to get xml file
    # OldCoordinatesLetters(r"C:\Users\emwil\Downloads\ey_146.xml", r"C:\Users\emwil\Downloads\einyitzchak146.png", r"C:\Users\emwil\Downloads\einyitzchak-146_boxes.png")
    # # ein yitzchok
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\ey_1.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-025.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\ey_2.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-032.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\ey_3.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-244.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\ey_4.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-247.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\ey_5.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-258.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\ey_6.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\einyitzchak-259.png")
    # #meishiv davar
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\md_1.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\mishivdavar-0156.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\md_2.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\mishivdavar-066.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\md_3.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\mishivdavar-037.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\md_4.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\mishivdavar-035.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\md_5.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\mishivdavar-031.png")
    # getCoordinates(r"C:\Users\emwil\Downloads\tests\md_6.xml", r"C:\Users\emwil\Downloads\Telegram Desktop\mishivdavar-030.png")

    # to copy files from one folder to another
    # move_files()

    # clean images
    # p = r"C:\Users\emwil\Downloads\processed_images\processed_images"
    # with os.scandir(p) as it:
    #     for entry in it:
    #         if entry.name.endswith(".tif") and entry.is_file():
    #             print("cleaning up: " + entry.name)
    #             input_p = entry.path
    #             # output_p = r"C:\Users\emwil\Downloads\processed_images\cleaned_images\sharpened" + os.sep + entry.name
    #             output_p = r"C:\Users\emwil\Downloads\processed_images\cleaned_images"
    #             clean_image(input_p, output_p, entry.name)
    # print("done cleaning")

    # ocr()

    # find_clumps(r"C:\Users\emwil\Downloads\two_cols.csv")


    # total_text = []
    # # # # print(df)
    chars = []
    counter = 0
    # # # # # # #new files
    # # # names = []
    # # # line_pos = []
    relative_height = []
    relative_width = []
    colon_aprox = []
    left = []
    right = []
    bottom = []
    top = []
    line_width = []
    line_height = []
    chars_per_line = []
    num_per_book = []
    line_num = []
    # path = r"C:\Users\emwil\Downloads\bad_xml"
    path = r"C:\Users\emwil\Downloads\processed_images_xml"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".xml") and entry.is_file():
    #             # if (entry.name != "masatbinyamin-013.tif_bold.xml"):
    #             #
    #             print(entry.name, entry.path)
                pic = entry.name.removesuffix(".xml")
                name = pic.removesuffix(".tif")
    #             # print("name: " + name)
    #             # names.append(name)
                pic_path = r"C:\Users\emwil\Downloads\bad_pics\low score images" + os.sep + pic
    #             # print("pic path: " + pic_path)
    #             output_path_name = r"C:\Users\emwil\Downloads\processed_images\bolded_processed_images" + os.sep + name + ".png"
                output_path_name = r"C:\Users\emwil\Downloads\bad_bolded" + os.sep + name + ".png"
                xml_path = entry.path
                count = 0
                # count = getCoordinates(xml_path, pic_path, output_path_name, name, chars, counter, count)
                # num_per_book.append([name, count])
    # print(entry.path, pic_path, output_path_name)
    #             relative_height, relative_width, colon_aprox = bolded_by_height(xml_path, pic_path,
    #                                                 output_path_name, relative_height, relative_width, colon_aprox)
    #             left, right, top, bottom = bolded_by_height(xml_path, pic_path, output_path_name, left, right, top, bottom)
                line_width, line_height, chars_per_line, line_num = add_feature(xml_path, line_width, line_height, chars_per_line, line_num)
                # bolded_by_width(xml_path, pic_path, output_path_name)
    #             t = getCoordinates(entry.path, pic_path, output_path_name, name, chars, counter)
    #             # print(t)
    #             total_text.append(t)

    # for i in range(6):
    #     print(left[i], right[i], top[i], bottom[i])
    # print(len(left), len(right), len(top), len(bottom))
    # df = pd.DataFrame(data=num_per_book, columns=['Book', 'Num of Charachters'])
    # df.to_csv(r"C:\Users\emwil\Downloads\Num_Chars.csv")
    df = pd.read_csv(r"C:\Users\emwil\cs_projects\BoldDetection\data.csv", encoding='utf-8-sig')
    print(len(df))
    print(len(line_width), len(line_height), len(chars_per_line), len(line_num))
    input("press enter if good: ")
    df['line width'] = line_width
    df['line height'] = line_height
    df['charachters per line'] = chars_per_line
    df['line number'] = line_num
    df.to_csv(r"C:\Users\emwil\cs_projects\BoldDetection\data.csv", encoding='utf-8-sig')
    # rel_size = []
    # for i in range(len(relative_height)):
    #     rel_size.append((relative_height[i] * relative_width[i]))
    # df['rel_size'] = rel_size
    # df['rel_height'] = relative_height
    # df['rel_width'] = relative_width
    # df['col_aprox'] = colon_aprox
    # df['rel_left'] = left
    # df['rel_right'] = right
    # df['rel_top'] = top
    # df['rel_bottom'] = bottom
    # df.to_csv(r"C:\Users\emwil\cs_projects\BoldDetection\data.csv", encoding='utf-8-sig')
    # Adding Column for Position of Charchter in Line
    # for i in line_pos:
    #     print(i)
    # print(len(line_pos))
    # df = pd.read_csv(r"C:\Users\emwil\Downloads\csv_data.csv", encoding='utf-8-sig')
    # print(len(df['height']))
    # df['Line Pos'] = line_pos
    # df.to_csv(r"C:\Users\emwil\Downloads\csv_data.csv", encoding='utf-8-sig')


    # for i in range(len(total_text)):
    #     print(i)
    #     print(str(total_text[i]))

    # for f in range(len(total_text)):
    #     folder = r"C:\Users\emwil\Downloads\abbyy_text"
    #     file_path = folder + os.sep + names[f] + ".txt"
    #     file = open(file_path, "w+", encoding='utf-8-sig')
    #     file.write(str(total_text[f]))
    #     file.close()


    # Loading list from getcoordinates into DataFrame
    # input("Press Enter to continue...")
    # df = pd.DataFrame(chars, columns=['char', 'Char Num', 'size', 'cors', 'height', 'width', 'ASCII val', 'Book Name', 'Page Num', 'Bolded'])
    # Order is: 1) Book Name, 2) Page Number, 3) Char, 4) Bolded, 5) Coordinates, 6) Size, 7) Height, 8) Width
    # 9) ASCII val, 10) counter, 11) Line Location, 12) Left, 13) Right, 14) Top, 15) Bottom
    # print(chars[3])
    # df = pd.DataFrame(chars, columns=['Book Name','Page Number','Char', 'Bolded', 'Coordinates', 'Size', 'Height', 'Width',
    #                                   'ASCII val','counter', 'Line Location','Left', 'Right', 'Top', 'Bottom'])
    # print(df.head(3))
    # df.to_csv(r'C:\Users\emwil\cs_projects\BoldDetection\data.csv', encoding='utf-8-sig')
    # print("loaded data frame")

    c_path = r'C:\Users\emwil\cs_projects\BoldDetection\data.csv'
    csv_path = r'C:\Users\emwil\Downloads\csv_data.csv'
    csv_1 = r"C:\Users\emwil\Downloads\csv_data_1.csv"

    # train(c_path)

    # csv_1 = r"C:\Users\emwil\Downloads\csv_data_1e.xlsx"
    # fill_bold(c_path)
    # ground_truth_bold(c_path)

    # ground_truth_bold(csv_path)
    # fill_bold(csv_1)
    # ground_truth_bold(csv_1)

    excel_path = r"C:\Users\emwil\Downloads\csv_data_excel.xlsb"
    # visualize(csv_path)



    #             # temp_path_name = r"C:\Users\emwil\Downloads\random images to send" + os.sep + name + ".png"
    #             # print(entry.path, pic_path, temp_path_name)
    #             # getCoordinates(entry.path, pic_path, temp_path_name)
    #
    # # has the following structure: [total_num_chars, char.text, let_size, x0_y0_x1_y1, h, w, let_num, n]
    # # df = pd.DataFrame(chars, columns=['Book Name   ', 'ASCII val', 'width', 'height', 'cors', 'size', 'char', 'Char Num'])
    # df = pd.DataFrame(chars, columns=['char', 'Char Num', 'size', 'cors', 'height', 'width', 'ASCII val', 'Book Name', 'Page Num', 'Bolded'])
    # # total_num_chars, char.text, let_size, x0_y0_x1_y1, h, w, let_num, n])
    # print(df)
    # print(df.describe)
    # # writer = pd.ExcelWriter(r"C:\Users\emwil\Downloads\excel_data.xlsx")
    # # encoding = 'utf-8-sig'
    # df.to_csv(r"C:\Users\emwil\Downloads\csv_data.csv", encoding = 'utf-8-sig')


    # data = pd.DataFrame(r"C:\Users\emwil\Downloads\csv_data.csv")
    # data = pd.read_csv(r"C:\Users\emwil\Downloads\csv_data.csv")
    # print(data.head)
    # print(total_text)
    # # mark * after bold charachters


    # write all text to file
    # print("total text:")
    # print(total_text)
    # file = open(r"C:\Users\emwil\Downloads\total_text.txt", 'w+', encoding = 'utf-8-sig')
    # file.write(total_text)
    # file.close()
    # input("Press Enter to continue...")

    # now read it back in
    # file = open(r"C:\Users\emwil\Downloads\tot_text.txt", 'r', encoding = 'utf-8-sig')
    # contents = file.read()
    # print(contents)
    # up_to_num = 143775
    # for i in contents:
    #     if (i == '.' or i == ':' or i == ' ' or i == '\n'):
    #         continue
    #     if (i == '*'):
    #         # mark previous index bold
    #         print("found bold at index: " + str(up_to_num -1))
    #     else:
    #         up_to_num += 1
    # df.loc[(df['age'] == 21) & df['favorite_color'].isin(array)]


    # data.loc[(data['Book Name'] == "masatbinyamin") & (data['Page Num'] == '13')]



    # d = data.groupby((data['Book Name'] == "masatbinyamin") & data['Page Num'] == '13')
    # print(d.head(1))
    # start = data.loc[(data['Book Name'] == "masatbinyamin") & (data['Page Num'] == "13")]
    # print(start)
    # df.to_excel(writer, encoding = 'utf-8-sig')
    # writer.save()
    # df.to_csv()
    # print(df.head)
    # print(df.tail)
    # for i in df:
    #     print(i)
    #
#   getCoordinates(r"C:\Users\emwil\Downloads\processed_images_xml\achiezer-007.tif_bold.xml", r"C:\Users\emwil\Downloads\processed_images\processed_images\achiezer-007.tif")
    # getCoordinates(r"C:\Users\emwil\Downloads\processed_images_xml\1.xml", r"C:\Users\emwil\Downloads\processed_images\processed_images\achiezer-007.tif")
    # ocr()

