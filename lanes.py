import cv2
import numpy as np



def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3.0 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    avg_left_fit = np.average(left_fit, axis=0)
    avg_right_fit = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, avg_left_fit)
    right_line = make_coordinates(image, avg_right_fit)
    return np.array([left_line, right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# '''canny edge  function '''

def canny(image):
    if image is None:
        road_cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def interset_region(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


try:
    road_cap = cv2.VideoCapture('test2.mp4')
    while (road_cap.isOpened()):
        _, frame = road_cap.read()
        canny_image = canny(frame)
        cropped_image = interset_region(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        average_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, average_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('display straight lane detection video', combo_image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    road_cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f' something wrong for playing video{e}')
