import cv2
import os  # path processing
import xml.etree.ElementTree as ET  # use for xml processing
import numpy as np


ground_truth_cnt = 0
true_cnt = 0

ngang = 0
doc = 0
Total_recall = 0
Total_precision = 0


def removeLine(image, thresh):
    # ========Begin remove line========
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    global ngang
    ngang = 0
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
        ngang = ngang + 1

    # Remove vertical
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    global doc
    doc = 0
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
        doc = doc + 1;

    # ========End remove line========


# Draw bounded box from .xml data
def getAnnotation(xml_path):
    tree = ET.parse(xml_path)

    root = tree.getroot()

    sample_annotations = []
    global ground_truth_cnt
    ground_truth_cnt = 0
    for neighbor in root.iter('bndbox'):
        xmin = int(neighbor.find('xmin').text)
        ymin = int(neighbor.find('ymin').text)
        xmax = int(neighbor.find('xmax').text)
        ymax = int(neighbor.find('ymax').text)

        ground_truth_cnt += 1
        sample_annotations.append((xmin, ymin, xmax, ymax))
    return sample_annotations

def drawIoUResult(img, sample_annotations, boundedBox, cnt):
    global true_cnt
    true_cnt = 0
    for bbox in sample_annotations:
        iou = 0
        for bbox_pending in boundedBox:
            x1 = bbox[0]
            y1 = bbox[1]
            w1 = bbox[2] - bbox[0]
            h1 = bbox[3] - bbox[1]

            x2 = bbox_pending[0]
            y2 = bbox_pending[1]
            w2 = bbox_pending[2] - bbox_pending[0]
            h2 = bbox_pending[3] - bbox_pending[1]

            if ((x1+w1 >= x2) and (x2+w2 >= x1) and (y1+h1 >= y2) and (y2+h2 >= y1)):

                # below are coordinates of interaction area
                x_left = max(bbox[0], bbox_pending[0])
                y_top = max(bbox[1], bbox_pending[1])
                x_right = min(bbox[2], bbox_pending[2])
                y_bottom = min(bbox[3], bbox_pending[3])
                # --

                intersection_area = (x_right - x_left) * (y_bottom - y_top)

                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                bbox_pending_area = (bbox_pending[2] - bbox_pending[0]) * (bbox_pending[3] - bbox_pending[1])

                iou = intersection_area / float(bbox_area + bbox_pending_area - intersection_area)


                # print(iou)  # print IoU of each cell
                if iou > 0.5:
                    true_cnt += 1
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        iou = round(iou, 2)
        text = str(iou)
        # print(iou)

        cv2.putText(img, text, (bbox[0] - 4, bbox[3] + 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        # print("-------")
    print("Number of Ground-truth cells: ", ground_truth_cnt)
    print("Number of correctly detected cells: ", true_cnt)
    print("Recall: ", round((true_cnt / ground_truth_cnt), 2))
    print("Precision: ", round((true_cnt / cnt), 2))

    global Total_precision
    Total_precision = Total_precision + (true_cnt / cnt)

    global Total_recall
    Total_recall = Total_recall + round((true_cnt / ground_truth_cnt), 2)
    print("------------")


def tableDetection(img_path, xml_path):
    img = cv2.imread(img_path)
    print(img_path)
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # remove line in table image
    removeLine(image, thresh)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, img_bino) = cv2.threshold(img_gray, 188, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bino = cv2.bitwise_not(img_bino)

    (thresh, img_bin) = cv2.threshold(img_gray, 188, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    (x1, y1, _) = img.shape

    # cv2.imshow("img", img_bin)
    # cv2.waitKey(0)

    (thresh, img_temp) = cv2.threshold(img_gray, 188, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_temp = cv2.bitwise_not(img_temp)
    for ii in range(0, x1):
        for jj in range(10, y1):
            if img_temp[ii][jj - 1] > 123:
                img_temp[ii][jj] = 255


    for ii in range(0, x1):
        for jj in reversed(range(1, y1 - 10)):
            if img_temp[ii][jj] > 123:
                img_temp[ii][jj - 1] = 255


    # cv2.imshow("bin1", img_temp)
    # cv2.waitKey(0)

    temp = 1
    count_line = 0
    total_line = 0
    space_line = []
    for ii in range(0, x1 - 1):
        if img_temp[ii][y1 - 1] == 0 and img_temp[ii + 1][y1 - 1] == 0:
            temp = temp + 1

        if img_temp[ii][y1 - 1] == 0 and img_temp[ii + 1][y1 - 1] != 0:
            count_line = count_line + 1
            total_line = total_line + temp
            space_line.append(temp)
            temp = 1

    # print ("spaceline", space_line)
    # print (max(space_line), min(space_line), round(sum(space_line)/len(space_line)))

    (thresh, img_tmp) = cv2.threshold(img_gray, 188, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_tmp = cv2.bitwise_not(img_tmp)
    for ii in range(1, x1):
        for jj in range(0, y1):
            if img_tmp[ii - 1][jj] > 123:
                img_tmp[ii][jj] = 255

    for ii in reversed(range(1, x1)):
        for jj in range(0, y1):
            if img_tmp[ii][jj] > 123:
                img_tmp[ii - 1][jj] = 255
    #
    # cv2.imshow("bin2", img_tmp)
    # cv2.waitKey(0)

    temp = 1
    count_row = 0
    total_row = 0
    space_row = []
    for jj in range(10, y1 - 1):
        if img_tmp[x1 - 1][jj] == 0 and img_tmp[x1 - 1][jj + 1] == 0:
            temp = temp + 1

        if img_tmp[x1 - 1][jj] == 0 and img_tmp[x1 - 1][jj + 1] != 0:
            count_row = count_row + 1
            space_row.append(temp)
            total_row = total_row + temp
            temp = 1

    # print("space_row", space_row)
    # print(max(space_row), min(space_row), round(sum(space_row) / len(space_row)))

    img_bin = img_temp & img_tmp
    #
    # cv2.imshow("bin3", img_bin)
    # cv2.waitKey(0)
    # cv2.imshow("bin4", img_bino)
    # cv2.waitKey(0)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(y1 / (doc + 2) / 8.5), round(x1 / (ngang + 2) / 5)))  # kernel width on the left, kernal height on the right

    kernel1 = 0
    kernel2 = 0
    Total = 0
    Ave = sum(space_line) / len(space_line)
    for temp in space_line:
        Total += abs(temp - Ave)

    nhan = 1.2
    if ngang > 1 and (len(space_line) / ngang) > 2:
        nhan = 2.1

    if len(space_line) != 0:
        kernel2 = round(Total * nhan/ len(space_line))
    # print ("Ke2", kernel2)

    Total = 0
    Ave = round(sum(space_row) / len(space_row))
    for temp in space_row:
        Total += abs(temp - Ave)
    if len(space_row) != 0:
        kernel1 = round(Total * nhan/ len(space_row))
    # print("Ke1", kernel1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel1 + 1, kernel2 + 1))

    # print (round(y1 / (doc + 2) / 8.5), round(x1 / (ngang + 2) / 5))
    connected = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("connected", connected)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(image=connected, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    cnt = 0
    boundedBox = []
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        #discard areas that are too large
        if h > x1 * 0.7 and w > y1 * 0.7:
            continue

        # discard areas that are too small
        if h < 4 or w < 4:
            continue

        xmax = 0
        ymax = 0
        xmin = 1000000
        ymin = 1000000
        check = 0
        for ii in range(0, w - 3):
            for jj in range(0, h):
                if img_bino[y + jj][ii + x] == 255:
                    check = 1
                    xmax = max (xmax, ii + x)
                    xmin = min (xmin, ii + x)
                    ymin = min (ymin, jj + y)
                    ymax = max (ymax, jj + y)

        if check == 0:
            continue
        # print(xmin, xmax, ymin, ymax)
        boundedBox.append((xmin - 1, ymin - 1, xmax + 2, ymax + 1))
        # draw rectangle around contour on original image
        cv2.rectangle(img, (xmin - 1, ymin - 1), (xmax + 2, ymax + 1), (255, 0, 0), 1)
        cnt += 1  # number of cells detected

    # print("Number of cells detected: ", cnt)
    print("------------")

    sample_annotations = getAnnotation(xml_path)
    drawIoUResult(img, sample_annotations, boundedBox, cnt)

    # plt.figure(figsize=(15, 15))
    # plt.imshow(img)
    # # cv2.imwrite("ans.jpg", img)
    # cv2.imshow("bin5", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


path_folder = "publictest"

for i in os.listdir('./' + path_folder + '/'):  # image store in "forms" folder
    if i.endswith('.png') or i.endswith('.PNG') or i.endswith('.jpg') or i.endswith('.JPG'):
        tableDetection(os.path.splitext('./' + path_folder + '/' + i)[0] + '.png',
                       os.path.splitext('./' + path_folder + '/' + i)[
                           0] + '.xml')  # os.path.splitext use for split file name out of extension

print("Average recall: ", Total_recall / (len(os.listdir('./' + path_folder + '/')) / 2))
print("Average precision: ", Total_precision / (len(os.listdir('./' + path_folder + '/')) / 2))