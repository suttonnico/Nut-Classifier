import cv2

N = 4
def zero_pad(x,n):
    for i in range(1,5):
        if x < 10 ** i:
            return (n-i)*'0'+str(x)


cam_1 = cv2.VideoCapture(1)   # 0 -> index of camera

cam_2 = cv2.VideoCapture(2)   # 0 -> index of camera
nut_dir = 'data/recinto'


for i in range(N):
    s, img_1 = cam_1.read()
    s, img_2 = cam_2.read()
    cv2.imwrite(nut_dir + '/nuez_' + zero_pad(i, 4) + '_1.png', img_1)
    cv2.imwrite(nut_dir + '/nuez_' + zero_pad(i, 4) + '_2.png', img_2)


Ariiba = '1 cm'