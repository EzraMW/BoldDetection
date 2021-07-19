from PIL import Image, ImageDraw, ImageEnhance
import pytesseract
import statistics
import os
from xml.etree import ElementTree as ET
from matplotlib import pyplot as plt
import pyxlsb
from IPython.display import display

import pickle
import nltk

import pandas as pd
import numpy as np
import cv2
import time


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

    # with open(img_2, 'rb') as f:
    #     data = f.read()
    #
    # with open('picture_out.png', 'wb') as f:
    #     f.write(data)

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

def getCoordinates(xml, png, output, n, chars, counter, line_loc):
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
                                    text_size_dim.append([char.text, let_size, x0_y0_x1_y1])
                                    total_letter_size += let_size
                                    # total_num_chars = len(c) + 1
                                    chars.append([char.text, counter, let_size, x0_y0_x1_y1, h, w, let_num, name, page_num, False])
                                    counter += 1
                                    line_loc.append(lc)
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
                # print("text: " + str(i[0]) + " size: " + str(i[1]))

        # img_2.show()
        # img_2.save(output, "PNG")

        # num of charachters
        # print("num of charachters in this doc: " + str(len(text_size_dim)) + " in " + n)
        # print("total num of charachters seen so far: " + str(len(c)))
        return text




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


def fill_bold(csv_path):
    print("loading in prev csv file")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # print("printing values:")
    # for i in range(1,9):
    #     print(i, df[7][i])

    # print(str(df['Bolded'][3922]))
    # print(str(df[7]['Book Name']))
    # print(str(df[7][9]))

    up_to_num = 0
    path = r"C:\Users\emwil\Downloads\abbyy_text"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".txt") and entry.is_file():
                print("up to file: " + entry.name)
                input("Press Enter when done adding stars...")
                file = open(entry.path, 'r', encoding='utf-8-sig')
                contents = file.read()
                # up_to_num = 143775
                for i in contents:
                    if (i == '.' or i == ':' or i == ' ' or i == '\n'):
                        continue
                    if (i == '~'):
                        # up to (and not including) ohelmosheresponsa-018.txt
                        # mark previous index bold
                        print("before " + str(df['Bolded'][up_to_num - 1]))
                        df['Bolded'][up_to_num - 1] = True
                        print("found bold at index: " + str(up_to_num - 1))
                        print("after " + str(df['Bolded'][up_to_num - 1]))
                    else:
                        up_to_num += 1
                df.to_csv(r'C:\Users\emwil\Downloads\csv_data.csv', encoding='utf-8-sig')

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




if __name__ == '__main__':

    # do stuff to get xml file

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



    # total_text = []
    # # # print(df)
    chars = []
    counter = 0
    # # # #new files
    names = []
    line_pos = []
    path = r"C:\Users\emwil\Downloads\processed_images_xml"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".xml") and entry.is_file():
                # if (entry.name == "masatbinyamin-013.tif_bold.xml"):
                # print(entry.name, entry.path)
                pic = entry.name.removesuffix("_bold.xml")
                name = pic.removesuffix(".tif")
                # print("name: " + name)
                # names.append(name)
                pic_path = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + pic
                # print("pic path: " + pic_path)
                output_path_name = r"C:\Users\emwil\Downloads\processed_images\bolded_processed_images" + os.sep + name + ".png"
                # print(entry.path, pic_path, output_path_name)
                t = getCoordinates(entry.path, pic_path, output_path_name, name, chars, counter, line_pos)
    #             total_text.append(t)

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



    # input("Press Enter to continue...")
    # df = pd.DataFrame(chars, columns=['char', 'Char Num', 'size', 'cors', 'height', 'width', 'ASCII val', 'Book Name', 'Page Num', 'Bolded'])
    # df.to_csv(r'C:\Users\emwil\Downloads\csv_data.csv', encoding='utf-8-sig')
    # print("loaded data frame")

    csv_path = r'C:\Users\emwil\Downloads\csv_data.csv'
    # fill_bold(csv_path)

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

