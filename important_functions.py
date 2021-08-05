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

# get text from images using tesseract OCR of an entire directory of images and write text files to output path
def get_text(directory_path, output, suffix):
    with os.scandir(directory_path) as it:
        for entry in it:
            if entry.name.endswith(suffix) and entry.is_file():
                print(entry.name)
                try:
                    text = pytesseract.image_to_string(entry.path, lang="heb_tess_dict")
                    name = entry.name.removesuffix(suffix)
                    output_path = output + os.sep + name + ".txt"
                    file = open(output_path, "w", encoding='utf-8-sig')
                    file.write(text)
                    file.close()
                except:
                    continue

# get a list of the coordinates of each word from the image, from it's xml form from ABBYY, and draw bounding boxes around
# each word
def get_coordinates_of_words(xml, png, output):
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
                            # seperate words by line
                            if (num_words != 0 and len(curr) != 0):
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
                                total_size += word_size
                                words.append(curr)
                                s = "".join(word)
                                all_info_words.append([x0_y0_x1_y1, word_size, s])
                                num_words += 1
                                curr = []
                                word = []
                            for lang in line:
                                for char in lang:
                                    # seperate words by spaces
                                    if char.text == ' ':
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
                                        total_size += word_size
                                        words.append(curr)
                                        s = "".join(word)
                                        all_info_words.append([x0_y0_x1_y1, word_size, s])
                                        num_words += 1
                                        curr = []
                                        word = []
                                    else:
                                        if (char.text == '.' or char.text == ':'):
                                            continue
                                        curr.append(char)
                                        word.append(char.text)
    img = Image.open(png).convert('RGBA')
    img_2 = img.copy()
    draw = ImageDraw.Draw(img_2)
    for i in all_info_words:
        draw.rectangle(i[0], outline="black", width=1)
    # img_2.show()
    img_2.save(output, "PNG")

# Get the coordinates of each word from the xml representation of the image and draw bounding boxes around the words
# Then, using the relative sizes of the words this algorithm attempts to predict whether the word is bold or not and
# labels that by drawing BLACK bounding boxes around regular words and RED around predicted Bolded ones.
# Bold words are predicted as being double the size of the average word size and this can be adjusted for more/less
# precise predictions
def bolded_coordinates_based_on_words(xml, png, output):
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
                            # new word at new line (excluding the first line)
                            if (num_words != 0 and len(curr) != 0):
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
                                total_size += word_size
                                words.append(curr)
                                s = "".join(word)
                                all_info_words.append([x0_y0_x1_y1, word_size, s])
                                num_words += 1
                                curr = []
                                word = []
                            for lang in line:
                                for char in lang:
                                    # new word by space
                                    if char.text == ' ':
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
                                        total_size += word_size
                                        words.append(curr)
                                        s = "".join(word)
                                        all_info_words.append([x0_y0_x1_y1, word_size, s])
                                        num_words += 1
                                        curr = []
                                        word = []
                                    else:
                                        if (char.text == '.' or char.text == ':'):
                                            continue
                                        curr.append(char)
                                        word.append(char.text)

    avg_size = total_size / num_words
    img = Image.open(png).convert('RGBA')
    img_2 = img.copy()
    draw = ImageDraw.Draw(img_2)
    for i in all_info_words:
        if (i[1] > 2 * avg_size):
            draw.rectangle(i[0], outline="black", width=2)
        else:
            draw.rectangle(i[0], outline="red", width=2)
    # img_2.show()
    img_2.save(output, "PNG")

# This is the default and main method to get the coordinates from the xml representation of the picture and use various
# charachteristics from the xml data to predict boldness (these charachteristics can be adjusted)
# Boldness is predicted based on individual character information and, specifically, when it is more than 3x bigger than
# the average size of that character (identified by its ASCII value)
# Additionally, this algorithm can be used to return important information about each character to add to the dataset
def get_coordinates(xml, png, output, n, chars, counter, count):
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

    # separate name and page number
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
                            # can choose to pass in total text variable to get a single string of the entire text of
                            # all of hte documents being looked at and then just un-comment the tot_text variables
                            # and return tot_text
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

    # create text files from abbyy xml
    # folder = r"C:\Users\emwil\Downloads\abbyy_text"
    # file_path = folder + os.sep + n + ".txt"
    # file = open(file_path, "w+", encoding='utf-8-sig')
    # file.write(text)
    # file.close()

    # draw boxes around bolded words
    if (letter_count == 0):
        print("Did not recognize ANY letters from this document")
    else:
        # print("total text so far: " + str(len(tot_text)))

        img = Image.open(png).convert('RGBA')
        img_2 = img.copy()
        draw = ImageDraw.Draw(img_2)
        # mean = statistics.mean(let_sizes)
        # std_dev = statistics.stdev(let_sizes)

        # print("total mean: " + str(mean))
        # print("total standard deviation: " + str(std_dev))

        # Determine the means and standard deviation for the sizes of each character orginized by ASCII value
        means = {}
        std_devs = {}
        for i in range(1, 20000):
            if (len(obj['size_' + str(i)]) > 1):
                means['mean' + str(i)] = statistics.mean(obj['size_' + str(i)])
                std_devs['stdev' + str(i)] = statistics.stdev(obj['size_' + str(i)])
            else:
                means['mean' + str(i)] = 0
                std_devs['stdev' + str(i)] = 0

        # print results
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

        # Draw bounding boxes for bolded and non-bolded characters based
        for i in text_size_dim:
            num = ord(i[0])
            if (means['mean' + str(num)] == 0 or std_devs['stdev'+str(num)] == 0):
                continue
            if (i[1] < (means['mean'+str(num)]+(3* std_devs['stdev'+str(num)]))):
                draw.rectangle(i[2], outline="black", width=1)
            else:
                draw.rectangle(i[2], outline="red", width=1)
                # print("letter: " + i[0])
                # print("actual size: " + str(i[1]))
                # print("mean + 3 std devs: " + str((means['mean'+str(num)]+(3* std_devs['stdev'+str(num)]))))
                # print("text: " + str(i[0]) + " size: " + str(i[1]))

        # img_2.show()
        img_2.save(output, "PNG")
        return chars

        # num of characters
        # print("num of characters in this doc: " + str(len(text_size_dim)) + " in " + n)
        # print("total num of characters seen so far: " + str(len(c)))
        # return text

# use this method to add features to the dataframe by passing in the xml and all the features you want to add as lists
# return the feature lists of data and add them to the dataframe
# This particular instance will generate lists of line_width, line_height, chars_per_line, and line_num and return them
# to be added to the dataframe but they can be replaced and adjusted to add any desired feature
def add_feature(xml_path, line_width, line_height, chars_per_line, line_num):
    xml = ET.parse(xml_path)
    root = xml.getroot()
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
                                        continue
                                    line_width.append(rel_line_wid)
                                    line_height.append(rel__line_high)
                                    line_chars += 1
                                    line_num.append(num_line)
    # add in chars per line from the last line of the page
    for i in range(line_chars):
        chars_per_line.append(line_chars)
    return line_width, line_height, chars_per_line, line_num


# Method to simply copy files/pictures from one directory (original_path) to another (new_out_path)
def move_files():
    original_path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images"
    with os.scandir(original_path) as it:
        for entry in it:
            if entry.name.endswith(".png") and entry.is_file():
                image = cv2.imread(entry.path)
                new_out_path = r"C:\Users\emwil\Downloads\processed_images\cleaned_images_png" + os.sep + entry.name
                cv2.imwrite(new_out_path, image)

# input the csv_path for an csv file of the predicted boldness of each indexed character and its coordinates to visualize
# the results of the trained model and see where and, ultimatly, why it went wrong and right.
def show_predictions(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    pic_bold_cors = []
    # base = 10
    b = 1
    for rows in df.itertuples():
        # print(rows)
        # print(rows[b])
        # print(rows[b])
        name = rows[b] + "-" + str(rows[b + 1]).zfill(3) + ".tif"
        # print(name)
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b + 1]).zfill(2) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        # cor = [rows[13], rows[15], rows[14], rows[16]]
        cor = rows[b+2].removeprefix('(').removesuffix(')').split(', ')
        c = []
        for i in cor:
            c.append(int(i))
        pic_bold_cors.append([pic, rows[b + 3], c])
    print(pic_bold_cors[3])
    input("press enter:")

    prev_pic = (pic_bold_cors[0])[0]
    output = prev_pic.replace("processed_images\processed_images", "predict pics")
    # print(output)
    # print(prev_pic)
    img = Image.open(prev_pic).convert('RGBA')
    draw = ImageDraw.Draw(img)
    # print("creted draw")
    for a in pic_bold_cors:
        # print(a)
        if a[0] != prev_pic:
            output = prev_pic.replace("processed_images\processed_images", "predict pics")
            img.save(output)
            img.close()
            print("Next Pic: " + a[0])
            img = Image.open(a[0]).convert('RGBA')
            prev_pic = a[0]
            draw = ImageDraw.Draw(img)
        # img = Image.open(a[0]).convert('RGBA')
        if (a[1] == True):
            # if (a[3] <= 10):
            #     draw.rectangle(a[2], outline='green', width=2)
            # else:
            draw.rectangle(a[2], outline='red', width=2)
        elif (a[1] == False):
            draw.rectangle(a[2], outline='black', width=1)
    output = prev_pic.replace("processed_images\processed_images", "predict pics")
    img.save(output)
    img.close()



# Method to automate establishing the ground truth for the boldness of every character in the inputted csv/excel file (csv_path)
# The goal is to minimize the tediousness of establishing ground truth of the boldness of a large number of characters
# 1) Have a directory of text files compiled from data obtained from the OCR output (the ABBYY xml) which can be created in
# the getCoordinates method
# 2) initialize the Boldness column of the database to be false
# 3) Iterate through each text file in order
# 4) Before the algorithm reads the text file, it will stop. The user should open up the picture and text file side by side
#    and type a tilda character (~) after each character which is bolded and press enter when done with current file
# 5) The algorithm will then iterate through the characters of the file, aligning the up_to_num with the corosponding row
#    of the dataset and if the character == '~' then it labels the previous character's Bolded column as True, else False
def fill_bold(csv_path):
    print("loading in prev csv file")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    list = [False] * 293030
    print(len(list))
    df['Bolded'] = list

    up_to_num = 0
    path = r"C:\Users\emwil\Downloads\abbyy_text"
    bool = True
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".txt") and entry.is_file():
                print("up to file: " + entry.name)
                input("Press Enter when done adding tilda's (~)...")
                file = open(entry.path, 'r', encoding='utf-8-sig')
                contents = file.read()
                for i in contents:
                    if (bool and up_to_num == 141709):
                        up_to_num -= 4
                        bool = False
                    if (i == '.' or i == ':' or i == ' ' or i == '\n'):
                        continue
                    if (i == '~'):
                        # mark previous index bold
                        df['Bolded'][up_to_num - 1] = True

                    else:
                        up_to_num += 1
                df.to_csv(csv_path, encoding='utf-8-sig')

# Using the dataframe, create images to visualize the ground truth of boldness of the characters to confirm that the data
# was inputted correctly
def visualize_ground_truth(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    prev_name = " "
    count = 0

    pic_bold_cors = []
    # set the right value for the index of the 'Book Name' (which arbitrarily shifts sometimes) to get the right index
    # for the rest of the features
    b = 6
    for rows in df.itertuples():
        # print(rows)
        # print(rows[b])
        name = rows[b] + "-" + str(rows[b+1]).zfill(3) + ".tif"
        # fill in irregular zeros for file name for these two books
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b+1]).zfill(2) + ".tif"
        if (rows[b] == "zikukindenura"):
            name = rows[b] + "-" + str(rows[b+1]) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        # first transform coordinates from string to tuples
        cor = rows[b+4].removeprefix('(').removesuffix(')').split(', ')
        c = []
        for i in cor:
            c.append(int(i))
        pic_bold_cors.append([pic, rows[b+3], c])
    print("done making list")
    # df.to_csv(path, encoding='utf-8-sig')

    # Iterate through the chracters, starting with the first picture, and open up the image to be drawn upon and draw
    # the bounding box as red/black, depending on truth value of 'Bolded'. When the next picture is reached, close and save
    # the previous picture to a new folder, and open up the previous one
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
        if (a[1] == True):
            draw.rectangle(a[2], outline='red', width=1)
        elif (a[1] == False):
            draw.rectangle(a[2], outline='black', width=1)
    output = prev_pic.replace("processed_images\processed_images", "Ground Truth Pics")
    # img.show()
    img.save(output)
    img.close()


# Using the basic framework of the visualize_ground_truth method - visualize the ground truth to find the two clumps that
# the Bolded HEIGHT splits into
def find_clumps(path):
    two_cols = pd.read_csv(path, encoding='utf-8-sig')
    pic_bold_cors_height = []
    # base = 10
    b = 2
    for rows in two_cols.itertuples():
        name = rows[b] + "-" + str(rows[b + 1]).zfill(3) + ".tif"
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b + 1]).zfill(2) + ".tif"
        if (rows[b] == "zikukindenura"):
            name = rows[b] + "-" + str(rows[b + 1]) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        cor = [rows[13], rows[15], rows[14], rows[16]]
        pic_bold_cors_height.append([pic, rows[b + 3], cor, rows[b + 15]])
    prev_pic = (pic_bold_cors_height[0])[0]
    output = prev_pic.replace("processed_images\processed_images", "height pics")
    img = Image.open(prev_pic).convert('RGBA')
    draw = ImageDraw.Draw(img)
    for a in pic_bold_cors_height:
        if a[0] != prev_pic:
            output = prev_pic.replace("processed_images\processed_images", "height pics")
            img.save(output)
            img.close()
            print("Next Pic: " + a[0])
            img = Image.open(a[0]).convert('RGBA')
            prev_pic = a[0]
            draw = ImageDraw.Draw(img)
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

# Using the basic framework of the visualize_ground_truth method - visualize the ground truth to find the two clumps that
# the bolded WIDTH splits into
def find_width(path):
    two_cols = pd.read_csv(path, encoding='utf-8-sig')
    pic_bold_cors_width = []
    # base = 10
    b = 4
    for rows in two_cols.itertuples():
        # print(rows)
        # print(rows[b])
        # input("press enter: ")
        name = rows[b] + "-" + str(rows[b + 1]).zfill(3) + ".tif"
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b + 1]).zfill(2) + ".tif"
        if (rows[b] == "zikukindenura"):
            name = rows[b] + "-" + str(rows[b + 1]) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        cor = [rows[15], rows[17], rows[16], rows[18]]
        pic_bold_cors_width.append([pic, rows[b + 3], cor, rows[27]])
    prev_pic = (pic_bold_cors_width[0])[0]
    output = prev_pic.replace("processed_images\processed_images", "width pics")
    img = Image.open(prev_pic).convert('RGBA')
    draw = ImageDraw.Draw(img)
    for a in pic_bold_cors_width:
        if a[0] != prev_pic:
            output = prev_pic.replace("processed_images\processed_images", "width pics")
            img.save(output)
            img.close()
            print("Next Pic: " + a[0])
            img = Image.open(a[0]).convert('RGBA')
            prev_pic = a[0]
            draw = ImageDraw.Draw(img)
        if (a[1] == True):
            if (a[3] <= 450):
                draw.rectangle(a[2], outline='green', width=2)
            elif (a[3] >= 550):
                draw.rectangle(a[2], outline='blue', width=2)
            else:
                draw.rectangle(a[2], outline='red', width=2)
        elif (a[1] == False):
            draw.rectangle(a[2], outline='black', width=1)
    output = prev_pic.replace("processed_images\processed_images", "width pics")
    img.save(output)
    img.close()

# Using the basic framework of the visualize_ground_truth method - visualize the ground truth to find the two clumps that
# the bolded NUMBER OF CHARACTERS PER LINE splits into
def find_line_chars(path):
    two_cols = pd.read_csv(path, encoding='utf-8-sig')
    pic_bold_cors_chars = []
    b = 4
    for rows in two_cols.itertuples():
        name = rows[b] + "-" + str(rows[b + 1]).zfill(3) + ".tif"
        if (rows[b] == "mishivdavar"):
            name = rows[b] + "-" + str(rows[b + 1]).zfill(2) + ".tif"
        if (rows[b] == "zikukindenura"):
            name = rows[b] + "-" + str(rows[b + 1]) + ".tif"
        pic = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + name
        cor = [rows[15], rows[17], rows[16], rows[18]]
        pic_bold_cors_chars.append([pic, rows[b + 3], cor, rows[29]])
    prev_pic = (pic_bold_cors_chars[0])[0]
    output = prev_pic.replace("processed_images\processed_images", "chars pics")
    img = Image.open(prev_pic).convert('RGBA')
    draw = ImageDraw.Draw(img)
    for a in pic_bold_cors_chars:
        if a[0] != prev_pic:
            output = prev_pic.replace("processed_images\processed_images", "chars pics")
            img.save(output)
            img.close()
            print("Next Pic: " + a[0])
            img = Image.open(a[0]).convert('RGBA')
            prev_pic = a[0]
            draw = ImageDraw.Draw(img)
        if (a[1] == True):
            if (a[3] > 15):
                draw.rectangle(a[2], outline='green', width=2)
            else:
                draw.rectangle(a[2], outline='red', width=2)
        elif (a[1] == False):
            draw.rectangle(a[2], outline='black', width=1)
    output = prev_pic.replace("processed_images\processed_images", "chars pics")
    img.save(output)
    img.close()



if __name__ == '__main__':
    # iterate through directory of xml files to create the dataframe, visualize the boldness, add features to the dataframe
    # or any of the functions possible

    # path = path to xml directory
    path = r"C:\Users\emwil\Downloads\processed_images_xml"
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".xml") and entry.is_file():
                pic = entry.name.removesuffix("_bold.xml")
                name = pic.removesuffix(".tif")
                pic_path = r"C:\Users\emwil\Downloads\processed_images\processed_images" + os.sep + pic
                output_path_name = r"C:\Users\emwil\Downloads\bolded_by_word" + os.sep + name + ".png"
                xml_path = entry.path
                count = 0
                # get_coordinates(xml_path, pic_path, output_path_name, name, chars)
                # add_features(...) etc.
