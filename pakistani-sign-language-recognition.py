from keras.preprocessing.sequence import pad_sequences
import datetime
import xlsxwriter
import math
import thread
from pykinect import nui
from Tkinter import *
import numpy    
import cv2
from PIL import Image, ImageTk
import imutils
import pygame
import pandas
import pickle

DEPTH_WINSIZE = 640, 480

screen_lock = thread.allocate()
tmp_s = pygame.Surface(DEPTH_WINSIZE, 0, 16)
   
start = False
predict = False

"""Class for Kinect Device, handles all tasks related to kinect"""

class Kinect:
    def __init__(self, App):
        self.kinect = None
        self.GUI = App
        self.data = []

    def depth_frame_ready(self, frame):
        """ depth frame handler function """
        with screen_lock:
            a = []

            frame.image.copy_bits(tmp_s._pixels_address)
            arr2d = (pygame.surfarray.array2d(tmp_s) >> 7 & 255)
            new_image = arr2d.astype(numpy.uint8)

            new_image = imutils.rotate_bound(new_image, 90)
            self.GUI.videoLoop(new_image)
            
            u = int(new_image[new_image>0].min()+8)
            l = int(new_image[new_image>0].min()+0)

            hand_segmented_image = cv2.inRange(new_image, l , u)
            hand_segmented_image = cv2.GaussianBlur(hand_segmented_image , (5,5) , 0)

            
            
            global start
            if not start:
                return

            #for hs in slice_img:
            _, contours, _ = cv2.findContours(hand_segmented_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
            area = 0
            area1 = len(hand_segmented_image[hand_segmented_image > 0])
            a.append(area1)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100 : #area less than threshold is considered noise
                    continue

                hull = cv2.convexHull(cnt, returnPoints = False)    #find convex hull for each contours

                defects = cv2.convexityDefects(cnt, hull)    #find defects in the hull
                if defects is None:
                    continue
                moments = cv2.moments(cnt)
                x,y = ((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))) #from moments find centre of the contours
                y = y+8        
                centre = (x,y)   
                v = []

                for i in range(defects.shape[0]):
                    s,e,f,d1 = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    if(start[1] < centre[1] + 50):
                        d = (math.sqrt((start[0]-centre[0])**2+(start[1]-centre[1])**2))
                        a1 = (start , far)
                        b1 = (end , far)
                        angle = self.ang(a1,b1)

                        cv2.line(hand_segmented_image,centre, start,[100,100,0],2)

                        a.append(d)                
                        a.append(angle)
                        a.append(d1)

            self.wrt(a)
            self.GUI.depthLoop(hand_segmented_image)

    def StartKinect(self):
        if self.kinect == None:
            self.kinect = nui.Runtime()
        else:
            print 'Kinect Is Already Open'

    def OpenDepthStream(self):
        self.kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)
        self.kinect.depth_frame_ready += self.depth_frame_ready

    def CloseKinectInstance(self):
        self.kinect.close()	
        self.kinect = None
        
    def create_file(self):
        #XLS Writer
        self.row = 1
        self.workbook = xlsxwriter.Workbook('data.xlsx')    # Create a workbook.
        self.worksheet = self.workbook.add_worksheet()            #add a worksheet
        #bold = self.workbook.add_format({'bold': 1})        # Add a bold format to use to highlight cells.
        #datetimeformat = workbook.add_format({'num_format': 'dd/mm/yy hh:mm:ss.0000000'})   # Add an Excel date format.
        #XLS Writer Variables end here        
    def xlswrite(self):    
        # Adjust the column width.
        self.worksheet.set_column(1, 1, 15)


    def wrt(self,x1):
        col = 0
        #worksheet.write_datetime(row, col ,  x1[0]  , datetimeformat )     

        for i in range(1,len(x1)):
            self.worksheet.write_string(self.row, col+i, str(x1[i]))

        self.row = self.row+1
    def cls(self):       
        self.workbook.close()
        
        
        #Finding Angle start
    def dot(self,vA, vB):
        return vA[0]*vB[0]+vA[1]*vB[1]
    
    def ang(self,lineA, lineB):
        # Get nicer vector form
        vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
        vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
        # Get dot prod
        dot_prod = self.dot(vA, vB)
        # Get magnitudes
        magA = self.dot(vA, vA)**0.5
        magB = self.dot(vB, vB)**0.5
        # Get cosine value
        cos_ = dot_prod/magA/magB
        # Get angle in radians and then convert to degrees
        angle = math.acos(dot_prod/magB/magA)
        # Basically doing angle <- angle mod 360
        ang_deg = math.degrees(angle)%360

        if ang_deg-180>=0:
            # As in if statement
            return 360 - ang_deg
        else:     
            return ang_deg

    #Finding angle end
    
class Translator:
    def pre_process(self):         
        pred = pandas.read_excel('data.xlsx', header = 0) 
        pred = pred.iloc[:,1:]
        pred = pred.fillna(0)
        
        pad = pad_sequences(pred.values, maxlen=51) 
        pad = numpy.rot90(pad)
        pad = pad_sequences(pad, maxlen=114, padding='post')
        pad = numpy.rot90(pad, k = -1)
        pad = pad.ravel()
        pad = pad.reshape(1, -1)
        return pad
    
    def predict(self):
        data = self.pre_process()
        filename = 'finalized_model_random_forest.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        r = loaded_model.predict(data)
        print loaded_model.predict_proba(data)
        w = ''
        if r == 0:
            w = 'Allah Hafiz'
        elif r == 3:
            w = 'Welcome'
        elif r == 1:
            w = 'Ap ka naam kiya ha'
        else:
            w = 'Ap kese hain'
        print w
        return w   
         
class App:
    def __init__(self, master):
        """
         Main window (master) configuration
        """
        self.kinectInstance = Kinect(self)
        self.cap = None
        self.thread = None
        self.stopEvent = None
        self.window = master
        self.Translator = Translator()
        master.title("Pakistani Sign Language Recognition System")
        width, height = master.maxsize()
        self.w, self.h = width, height
        master.geometry("%dx%d+0+0" % (width-5,height-60))
        master.resizable(width=False, height=False)

        """
        Top Header, contains uni name and project name
        """
        Background = Frame(master, bg='#666666', width = width, height= height)
        Background.pack()

        Header = Frame(Background, bg='#0099FF', width = width, height= 100)
        Header.pack(padx = 10)

        photo = PhotoImage(file="uni.gif")
        uni_logo = Label(Header, image=photo, bg='#0099FF')
        uni_logo.photo = photo
        uni_logo.pack(side = LEFT, padx = 50)

        photo1 = PhotoImage(file="psl.gif")
        psl_logo = Label(Header, image=photo1, bg='#0099FF')
        psl_logo.photo = photo1
        psl_logo.pack(side = RIGHT, padx = 50)

        uni_name = Label(Header, bg='#0099FF', fg='#FFFFFF',text="FAST National University of Computer & Emerging Sciences", font='CenturyGothic 16 bold', width = width)
        uni_name.pack()

        psl_name = Label(Header, bg='#0099FF',fg='#FFFFFF', text="Pakistani Sign Language Recognition System", font=("Century Gothic", 14))
        psl_name.pack()

        Centre = Frame(Background, bg='#666666', width = (width), height= height)
        Centre.pack()

        self.panel = None #to display video

        self.Video = Frame(Centre, bg='#FFFFFF', width = (width/2), height= height)
        self.Video.pack(side = LEFT, padx = 10, pady = 10)

        self.skel = None #to display skeleton

        self.Skeleton = Frame(Centre, bg='#FFFFFF', width = width, height = 450)
        self.Skeleton.pack(side = TOP, padx = 10, pady = 10)

        Right_Box = Frame(Centre, bg='#666666', width = width, height = height)
        Right_Box.pack(side = BOTTOM, padx = 10, pady = 10)

        Text = Frame(Right_Box, bg='#FFFFFF', width = width, height = 120)
        Text.pack(side = TOP, pady = 3)
        self.text = StringVar()
        self.text.set('...')
        self.translation = Label(Text, bg='#FFFFFF', fg='#000000', textvariable= self.text, font='CenturyGothic 16 bold', width = width, height = 7)
        self.translation.pack()

        Footer = Frame(Right_Box, bg='#666666', width = width, height= 100)
        Footer.pack(side = BOTTOM, anchor = SE)

        Start = Button(Footer, bg= '#0066CC', text = 'START', font='CenturyGothic 12 bold', command = self.CollectData)
        Start.pack(side  = LEFT)

        Pause = Button(Footer, bg= '#CCCC33', text = 'END', font='CenturyGothic 12 bold', command = self.End)
        Pause.pack(side  = LEFT, padx = 20)

        Quit = Button(Footer, bg= '#FF0000', text = 'PREDICT', font='CenturyGothic 12 bold', command = self.Predict)
        Quit.pack()
        
        self.StartVideo()
    def CollectData(self):
        self.kinectInstance.create_file()
        global start
        start = True
        
    def End(self):
        global start
        start = False
        self.kinectInstance.cls()

    def Predict(self):
        r = self.Translator.predict()
        self.text.set(r)
        pygame.mixer.init()
        path  = r+'.mp3'
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    def StartVideo(self):
        self.kinectInstance.StartKinect()
        self.kinectInstance.OpenDepthStream()
        

    def videoLoop(self, image):
        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format
        image = cv2.resize(image, (700, 450))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # if the panel is None, we need to initialize it
        if self.panel is None:
            self.panel = Label(self.Video, image=image)
            self.panel.image = image
            self.panel.pack()

        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def depthLoop(self, image):

        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format
        image = cv2.resize(image, (700, 450))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)


        # if the panel is None, we need to initialize it
        if self.skel is None:
            self.skel = Label(self.Skeleton, image=image)
            self.skel.image = image
            self.skel.pack()

        # otherwise, simply update the panel
        else:
            self.skel.configure(image=image)
            self.skel.image = image

root = Tk()
app = App(root)
root.mainloop()