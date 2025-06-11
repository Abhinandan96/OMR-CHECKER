from flask import Flask, render_template, request
import os
import cv2 as cv
import numpy as np
import functions
import uuid

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'omr_image' not in request.files:
            return "No file part"
        file = request.files['omr_image']
        if file.filename == '':
            return "No selected file"
        
        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result_img_path, score = process_omr(file_path)

        return render_template('result.html', result_img=result_img_path, score=score)
    
    return render_template('index.html')


def process_omr(image_path):
    question, options = 5, 5
    myAnswers = [1, 2, 0, 4, 4]
    choiceArea = 1000
    img_width, img_height = 550, 600

    image = cv.imread(image_path)
    image = cv.resize(image, (img_width, img_height))
    img_cont_2 = image.copy()
    Canny, Erode = functions.preprocessing(image)

    try:
        contours, img_cont = functions.get_contour(image, Canny, filters=4, draw=True, minArea=9000)
        cv.drawContours(img_cont_2, contours[0][0], -1, (0, 0, 255), 5)
        if len(contours) == 2:
            cv.drawContours(img_cont_2, contours[1][0], -1, (255, 0, 0), 5)
        BiggestContour = functions.reorder(contours[0][3])
        if len(contours) == 2:
            globalContour = functions.reorder(contours[1][3])
        pt1 = np.float32(BiggestContour)
        warpImg1 = functions.get_warp(image, pt1, img_width, img_height, img_width, img_height)
        warpCanny, warpErode = functions.preprocessing(warpImg1)
        contours_circle, img_cont_circle = functions.get_contour_circle(warpImg1, warpCanny, draw=True, minArea=choiceArea)
    except:
        contours_circle, img_cont_circle = functions.get_contour_circle(image, Canny, draw=True, minArea=choiceArea)

    x, y, points_c = [], [], []
    for obj in contours_circle:
        x.append(obj[3][0])
        y.append(obj[3][1])
        final_point = obj[3][0] + obj[3][2], obj[3][1] + obj[3][3]
        points_c.append(final_point)
    first = np.min(x), np.min(y)
    points_c = np.array(points_c)
    sum1 = points_c.sum(1)
    last = points_c[np.argmax(sum1)]

    warpImg = image.copy()[first[1] - 10:last[1] + 10, first[0] - 10:last[0] + 10]
    warpImg = cv.resize(warpImg, (img_width, img_height))
    img_cont_circle = img_cont_circle[first[1] - 10:last[1] + 10, first[0] - 10:last[0] + 10]
    img_cont_circle = cv.resize(img_cont_circle, (img_width, img_height))

    warpGray = cv.cvtColor(img_cont_circle, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(warpGray, 150, 255, cv.THRESH_BINARY_INV)[1]
    boxes = functions.splitting(threshold, question, options)

    myPixelValues = np.zeros((question, options), np.int32)
    countC, countR = 0, 0
    for x in boxes:
        totalPixels = cv.countNonZero(x)
        myPixelValues[countR][countC] = totalPixels
        countC += 1
        if countC == options:
            countR += 1
            countC = 0

    myIndex = [np.argmax(myPixelValues[x]) for x in range(0, question)]
    grading = [1 if myAnswers[x] == myIndex[x] else 0 for x in range(0, question)]
    score = int((sum(grading) / question) * 100)

    imgResult = warpImg.copy()
    imgResult = functions.show_answers(imgResult, myIndex, myAnswers, grading, question, options)
    # imgResult = functions.showAnswers(imgResult, myIndex, myAnswers, grading, questions=5, choices=5)


    result_path = os.path.join(UPLOAD_FOLDER, 'result_' + os.path.basename(image_path))
    cv.imwrite(result_path, imgResult)

    return result_path, score

if __name__ == '__main__':
    app.run(debug=True)
