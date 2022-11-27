import cv2
import numpy as np
def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("2", thresh)
    cv2.waitKey(0)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imshow("1", img_erode)
    cv2.waitKey(0)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("0", img_erode)
    cv2.waitKey(0)
    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    cv2.imshow("0", letters[0][2])
    cv2.imshow("1", letters[1][2])
    cv2.imshow("2", letters[2][2])
    cv2.imshow("3", letters[3][2])
    cv2.imshow("4", letters[4][2])
    cv2.imshow("5", letters[5][2])
    cv2.imshow("6", letters[6][2])
    cv2.imshow("7", letters[7][2])
    cv2.imshow("8", letters[8][2])
    cv2.imshow("9", letters[9][2])
    cv2.imshow("10", letters[10][2])
    cv2.waitKey(0)
    return letters
letters_extract('save2.png')