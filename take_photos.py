import cv2

N = 4
def zero_pad(x,n):
    for i in range(1,5):
        if x < 10 ** i:
            return (n-i)*'0'+str(x)


cam_1 = cv2.VideoCapture(0)   # 0 -> index of camera

cam_2 = cv2.VideoCapture(1)   # 0 -> index of camera
nut_dir = 'data/recinto'
class camera_man:
    i = 0
    def __init__(self,i):
        self.i = i
    def shoot(self):
        s, img_1 = cam_1.read()
        s, img_2 = cam_2.read()
        cv2.imwrite('recinto1_L/nuez_' + zero_pad(self.i, 6) + '.png', img_1)
        cv2.imwrite('recinto1_R/nuez_' + zero_pad(self.i, 6) + '.png', img_2)
        self.i = self.i +1


Ariiba = '1 cm'