from __future__ import print_function
import cv2
import sys
import math
import numpy as np
import cv2 as cv
import random

obrazok = "images/image61702.jpg"

img = cv2.imread(obrazok, cv2.IMREAD_UNCHANGED)

"""FILTROVANIE ZELENEJ"""

# It converts the BGR color space of image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold of green in HSV space
lower_green = np.array([40, 50, 40])
upper_green = np.array([70, 255, 255])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_green, upper_green)
cv.imwrite("images/mask.png", mask)
cv2.imshow('mask', mask)

"""VYPLNENIE MEDZIER"""

kernel = np.ones((5, 5), np.uint8)
dilation = cv.dilate(mask, kernel, iterations=5)

# calculate moments of binary image
M = cv2.moments(dilation)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(dilation, (cX, cY), 5, (120, 100, 20), -1)
cv2.putText(dilation, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 100, 20), 2)

cv2.imshow('dilation', dilation)
cv.imwrite("images/dilation.png", dilation)

"""VYSTRIHNUTIE IHRISKA"""

contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
mask2 = np.zeros(img.shape, np.uint8)

c = max(contours, key=cv2.contourArea)
# cv2.drawContours(mask2, c, -1, color=(255, 255, 255), thickness=cv2.FILLED)
cv2.fillPoly(mask2, [c], color=(255, 255, 255))

# erode mask
kernel = np.ones((5, 5), np.uint8)
mask2 = cv.erode(mask2, kernel, iterations=3)

# Show in a window
cv.imwrite("images/mask2.png", mask2)
cv2.imshow('mask2', mask2)

# VYSTRIHNUTIE Z POVODNEHO OBRAZKU
playground = cv2.bitwise_and(img, mask2)

cv.imwrite("images/playground.png", playground)
cv2.imshow('playground', playground)

"""DETEKCIA CIAR"""

edges = cv.Canny(playground, 50, 320, None, 3)

cv.imwrite("images/edges.png", edges)
cv2.imshow('edges', edges)

hough = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

linesP = cv.HoughLinesP(edges, 1, np.pi / 180,
                        30, None, 90, 35)  # threshold, None, minLineLen, maxLineGap
onlyLines = np.zeros((img.shape[0], img.shape[1]), np.uint8)

if linesP is None:
    print("v Hough Transform neboli najdene ziadne usecky!")
    sys.exit(1)

for i in range(0, len(linesP)):
    l = linesP[i][0]
    cv.line(hough, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
    cv.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 7, cv.LINE_AA)

cv.imwrite("images/hough.png", hough)
cv2.imshow('hough', hough)

"""ODSTRANENIE BLIZKYCH ROVNOBEZIEK (SKELETON)"""

# https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331

# TODO iny pristup (partition)
#   https://stackoverflow.com/questions/30746327/get-a-single-line-representation-for-multiple-close-by-lines-clustered-together

ret, thresh = cv.threshold(onlyLines, 127, 255, cv.THRESH_BINARY)

skel = np.zeros(thresh.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

while True:
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    temp = cv2.subtract(thresh, open)
    eroded = cv2.erode(thresh, element)
    skel = cv2.bitwise_or(skel, temp)
    thresh = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, oke (199oke (199quit the loop
    if cv2.countNonZero(thresh) == 0:
        break

cv.imwrite("images/skel.png", skel)
cv2.imshow('skel', skel)

"""DETEKCIA CIAR 2"""

# def line_point_distance(l, p):
#     # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
#     #   0  1  2  3        0  1
#     # l(x1,y1,x2,y2)    p(x0,y0)
#     cit = (l[2] - l[0]) * (l[1] - p[1]) - (l[0] - p[0]) * (l[3] - l[1])
#     men = (l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2
#     return abs(cit) / math.sqrt(men)


def linesegment_point_distance(l, p):
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    v = (l[0], l[1])
    w = (l[2], l[3])
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0:
        # v == w case
        return (v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2
    # Consider the line extending the segment, parameterized as v + t (w - v).
    # We find projection of point p onto the line.
    # It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # We clamp t from [0,1] to handle points outside the segment vw.
    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
    t = max(0, min(1, t))
    q = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    d_sqr = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
    return math.sqrt(d_sqr)


def podobnost(l1, l2):
    # sprav vektory
    u = (l1[2] - l1[0], l1[3] - l1[1])
    v = (l2[2] - l2[0], l2[3] - l2[1])
    # dlzky vektorov
    len1 = math.sqrt(u[0] ** 2 + u[1] ** 2)
    len2 = math.sqrt(v[0] ** 2 + v[1] ** 2)
    # test uhlov
    u_dot_v = u[0] * v[0] + u[1] * v[1]
    s = u_dot_v / (len1 * len2)
    if s > 1:
        s = 1
    theta = math.acos(s)
    # test vzdialenosti
    d1 = linesegment_point_distance(l1, (l2[0], l2[1]))
    d2 = linesegment_point_distance(l1, (l2[2], l2[3]))
    d3 = linesegment_point_distance(l2, (l1[0], l1[1]))
    d4 = linesegment_point_distance(l2, (l1[2], l1[3]))
    d = min([d1, d2, d3, d4])
    return s, d


def spoj(l1, l2, prahAlfa, prahD):
    s, d = podobnost(l1, l2)
    if abs(s) < np.cos(prahAlfa):
        return None
    if d > prahD:
        return None
    # spajaj ciary
    if l1[0] < l1[2]:
        lavy1 = (l1[0], l1[1])
        pravy1 = (l1[2], l1[3])
    else:
        lavy1 = (l1[2], l1[3])
        pravy1 = (l1[0], l1[1])
    if l2[0] < l2[2]:
        lavy2 = (l2[0], l2[1])
        pravy2 = (l2[2], l2[3])
    else:
        lavy2 = (l2[2], l2[3])
        pravy2 = (l2[0], l2[1])
    if lavy1[0] < lavy2[0]:
        lavy = lavy1
    else:
        lavy = lavy2
    if pravy1[0] > pravy2[0]:
        pravy = pravy1
    else:
        pravy = pravy2
    return [lavy[0], lavy[1], pravy[0], pravy[1]]


linesP = cv.HoughLinesP(skel, 1, np.pi / 180, 30, None, 90, 30)  # threshold, None, minLineLen, maxLineGap
if linesP is None:
    print("v Hough Transform neboli najdene ziadne usecky!")
    sys.exit(1)


def spajaj(linesP):
    ciary = []
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        ciary.append([l[0], l[1], l[2], l[3]])

    print("pocet pred:", len(ciary))

    spajanie = True
    while spajanie:
        spajanie = False
        nove = []
        for i in range(0, len(ciary)):
            for j in range(0, len(ciary)):
                if i == j:
                    continue
                l1 = ciary[i]
                l2 = ciary[j]
                if len(l1) > 4 or len(l2) > 4:
                    continue
                nova = spoj(l1, l2, np.pi / 50, 30)
                if nova is not None:
                    nove.append(nova)
                    l1.append(True)
                    l2.append(True)
                    spajanie = True
        for i in range(0, len(ciary)):
            l = ciary[i]
            if len(l) > 4:
                continue
            nove.append(l)

        ciary = nove

    print("pocet po:", len(ciary))
    return ciary


ciary = spajaj(linesP)

hough2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
for i in range(0, len(ciary)):
    l = ciary[i]
    rgb = (255, 255, 255)
    cv.line(hough2, (l[0], l[1]), (l[2], l[3]), rgb, 3, cv.LINE_AA)

# cv2.circle(hough2, (cX, cY), 5, (120, 100, 20), -1)
# cv2.putText(hough2, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 100, 20), 2)

cv.imwrite("images/hough2.png", hough2)
cv2.imshow('hough2', hough2)

### ZACIATOK NEPOUZITEHO POKUSU

"""HLADANIE NAJVZDIALENEJSICH CIAR"""

"""
pocUsekov = 8
kvant = []
useky = []
uhol = 0
for i in range(pocUsekov):
    kvant.append({'dlzka': 0})
    useky.append(uhol)
    uhol += 360//pocUsekov
useky.append(uhol)

perp = cv.cvtColor(skel, cv.COLOR_GRAY2BGR)
for i in range(0, len(ciary)):
    # povodna ciara
    l = ciary[i]
    rgb = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    cv.line(perp, (l[0], l[1]), (l[2], l[3]), rgb, 3, cv.LINE_AA)

    # kolmica
    aX, aY, bX, bY = l
    t = ((cX - aX)*(bX - aX) + (cY - aY)*(bY - aY)) / ((bX - aX)**2 + (bY - aY)**2)
    dX = int(aX + t*(bX - aX))
    dY = int(aY + t*(bY - aY))
    cv.line(perp, (cX, cY), (dX, dY), rgb, 1, cv.LINE_AA)

    # odmeraj a zakresli
    dlzka = math.sqrt((cX-dX)**2 + (cY-dY)**2)
    uhol = np.rad2deg(math.atan2(cY-dY, -(cX-dX)))
    if uhol < 0:
        uhol = 360 + uhol
    cv2.putText(perp, "dlzka:" + str(int(dlzka)), (dX - 25, dY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
    cv2.putText(perp, "uhol:" + str(int(uhol)), (dX - 25, dY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)

    # kvantizuj
    for j in range(len(kvant)):
        if uhol > useky[j] and uhol <= useky[j+1]:
            if dlzka > kvant[j]['dlzka']:
                kvant[j]['usek'] = str(useky[j]) + '..' + str(useky[j+1])
                kvant[j]['dlzka'] = dlzka
                kvant[j]['uhol'] = uhol
                kvant[j]['ciara'] = l
                kvant[j]['kolmica'] = (cX, cY, dX, dY)
                kvant[j]['rgb'] = rgb

cv.imwrite("images/prep.png", perp)
cv2.imshow('perp', perp)

print(kvant)

# vyber = cv.cvtColor(skel, cv.COLOR_GRAY2BGR)
vyber = np.zeros(img.shape, np.uint8)
for kv in kvant:
    if 'ciara' not in kv:
        continue
    rgb = kv['rgb']
    # ciara
    l = kv['ciara']
    cv.line(vyber, (l[0], l[1]), (l[2], l[3]), rgb, 3, cv.LINE_AA)
    # kolmica
    if False:
        k = kv['kolmica']
        cv.line(vyber, (k[0], k[1]), (k[2], k[3]), rgb, 1, cv.LINE_AA)
        cv2.putText(vyber, "dlzka:" + str(int(kv['dlzka'])), (k[2] - 25, k[3] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
        cv2.putText(vyber, "uhol:" + str(int(kv['uhol'])), (k[2] - 25, k[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)


cv.imwrite("images/vyber.png", vyber)
cv2.imshow('vyber', vyber)



### KONIEC NEVYUZITEHO POKUSU
"""

"""CONVEX HULL"""

# https://stackoverflow.com/questions/52197918/how-to-obtain-combined-convex-hull-of-multiple-separate-shapes
ret, thresh2 = cv.threshold(hough2, 127, 255, cv.THRESH_BINARY)
contours2, _ = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# spoj kontury do jednej
cont = np.vstack(contours2[i] for i in range(len(contours2)))
hull = cv.convexHull(cont)
conv_hull = np.zeros((hough2.shape[0], hough2.shape[1], 3), dtype=np.uint8)
# cv.drawContours(conv_hull, contours2, -1, (0,0,255))
# cv.drawContours(conv_hull, [hull], -1, (255,0,0))
cv2.fillPoly(conv_hull, [hull], color=(255, 255, 255))

cv.imwrite("images/conv_hull.png", conv_hull)
cv2.imshow('conv_hull', conv_hull)

"""VYBER OKRAJOVYCH CIAR"""

edges2 = cv.Canny(conv_hull, 50, 320, None, 3)

cv.imwrite("images/edges2.png", edges2)
cv2.imshow('edges2', edges2)

hough3 = np.zeros(img.shape, np.uint8)
linesP = cv.HoughLinesP(edges2, 1, np.pi / 180,
                        30, None, 90, 30)  # threshold, None, minLineLen, maxLineGap

if linesP is None:
    print("v Hough Transform neboli najdene ziadne usecky!")
    sys.exit(1)

linesP = spajaj(linesP)

for i in range(0, len(ciary)):
    l = ciary[i]
    # print(l, 'ciara')
    rgb = (0, 255, 0)
    cv.line(hough3, (l[0], l[1]), (l[2], l[3]), rgb, 3, cv.LINE_AA)

for i in range(0, len(linesP)):
    l = linesP[i]
    # print(l, 'linesP')
    rgb = (0, 0, 255)
    cv.line(hough3, (l[0], l[1]), (l[2], l[3]), rgb, 3, cv.LINE_AA)

prienik = []

DIST = 100
for i in range(0, len(ciary)):
    l1 = ciary[i]
    for j in range(0, len(linesP)):
        l2 = linesP[j]
        # print(l1, l2)
        # porovnaj
        s, d = podobnost(l1, l2)
        if abs(s) < np.cos(np.pi / 50):
            continue
        if (abs(l1[0] - l2[0]) < DIST and abs(l1[1] - l2[1]) < DIST and abs(l1[2] - l2[2]) < DIST and abs(
                l1[3] - l2[3]) < DIST) \
                or (abs(l1[0] - l2[2]) < DIST and abs(l1[1] - l2[3]) < DIST and abs(l1[2] - l2[0]) < DIST and abs(
            l1[3] - l2[1]) < DIST):
            prienik.append(l1)
            cv.line(hough3, (l1[0], l1[1]), (l1[2], l1[3]), (255, 0, 0), 3, cv.LINE_AA)

cv.imwrite("images/prienik.png", hough3)
cv2.imshow('prienik', hough3)

"""VYPOCET PRIESECNIKOV"""


def priesecnik(l1, l2):
    line1 = ((l1[0], l1[1]), (l1[2], l1[3]))
    line2 = ((l2[0], l2[1]), (l2[2], l2[3]))
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


priesecniky = np.zeros(img.shape, np.uint8)

for i in range(0, len(prienik)):
    l = prienik[i]
    # print(l)
    rgb = (255, 0, 0)
    cv.line(priesecniky, (l[0], l[1]), (l[2], l[3]), rgb, 3, cv.LINE_AA)

# TODO toto este nejako vymysliet; zatial aspon takto
# TODO: 0 a 1; 0 a 2; 2 a 3; 1 a 3
B1 = priesecnik(prienik[0], prienik[1])
B2 = priesecnik(prienik[0], prienik[2])
B3 = priesecnik(prienik[2], prienik[3])
B4 = priesecnik(prienik[1], prienik[3])
print(B1, B2, B3, B4)

cv.circle(priesecniky, (int(B3[0]), int(B3[1])), 5, (0, 255, 0), 2)
cv.circle(priesecniky, (int(B4[0]), int(B4[1])), 5, (0, 255, 0), 2)

cv.imwrite("images/priesecniky.png", priesecniky)
cv2.imshow('priesecniky', priesecniky)

"""HOMOGRAFIA"""

# 1000x637
body_vzor = np.float32([(0, 0), (1000, 0), (1000, 637), (0, 637)]).reshape(-1, 1, 2)
body_obrazok = np.float32([B1, B2, B3, B4]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(body_vzor, body_obrazok, cv2.RANSAC, 5.0)
print('H = ', H)

pts1 = np.float32([[250, 0], [530, 0], [530, 40], [250, 40]]).reshape(-1, 1, 2)  # lavy kraj
pts2 = np.float32([[250, 328], [630, 328], [630, 428], [250, 428]]).reshape(-1, 1, 2)  # stred
pts3 = np.float32([[350, 617], [510, 617], [510, 637], [350, 637]]).reshape(-1, 1, 2)  # pravy kraj
# pts1 = np.float32([[-150, -380], [-380, -380], [-330, -40], [-150, -40]]).reshape(-1, 1, 2)  # lavy kraj
# pts2 = np.float32([[40, 40], [40, 50], [50, 50], [50, 247]]).reshape(-1, 1, 2)  # stred

# pts3 = np.float32([[350, 617], [510, 617], [510, 637], [350, 637]]).reshape(-1, 1, 2)  # pravy kraj


pts4 = cv2.perspectiveTransform(pts1, H)
pts5 = cv2.perspectiveTransform(pts2, H)
pts6 = cv2.perspectiveTransform(pts3, H)
# dokresli transformovane suradnice do obrazku
vysledok = cv2.polylines(img, [np.int32(pts4)], True, 255, 3, cv2.LINE_AA)
# vysledok = cv2.polylines(img, [np.int32(pts5)], True, 255, 3, cv2.LINE_AA)
# vysledok = cv2.polylines(img, [np.int32(pts6)], True, 255, 3, cv2.LINE_AA)

cv.imwrite("images/vysledok.png", vysledok)
cv2.imshow('vysledok', vysledok)


cv2.waitKey(0)
cv2.destroyAllWindows()
