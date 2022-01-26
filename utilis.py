"""
https://theailearner.com/tag/cv2-warpperspective/
"""
import os
import csv
import xlwt
from xlwt import Workbook
import cv2
import numpy as np
import pandas as pd
import xlrd
import random
from imutils import build_montages




debug = True
#debug = False


paperSize = [510, 700]

solutios = {}

def	evaluateAns(ans, stdName, stdTemplate):
	global solutios
	# COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
	myIndex = getTemAns(ans, stdName, stdTemplate)

	grading=[]
	for x in range(0,len(ans)):
		
		if ans[x] == myIndex[x]:
			grading.append(1)
		elif ans[x] == '9':
			grading.append(9)
		else:
			grading.append(0)

	return grading, myIndex


def getMarkedCells(boxes, nr, nc):

	countR=0
	countC=0
	myPixelVal = np.zeros((nr,nc)) # TO STORE THE NON ZERO VALUES OF EACH BOX
	
	for image in boxes:
		myPixelVal[countR][countC]= cv2.countNonZero(image)
		countC += 1
		if (countC == nc):
			countC=0;
			countR +=1
	return myPixelVal

def getMarked(MarkedCells):
	x = np.argmax(MarkedCells, axis = 1)
	y = np.max(MarkedCells, axis = 1)
	s =""
	for i, z in enumerate(x):
		lst = MarkedCells[i]
		
		larger_elements = [element for element in lst if element > (y[i] -  50)]
		cnt = len(larger_elements)

		if cnt == 1:
			s = s + str(z)
		else:
			s = s + '9'
	return s


def wrap(img, partPoints, widthImg, heightImg, pad):

	start_point = (partPoints[0][0][0], partPoints[0][0][1])
	end_point = (partPoints[-1][0][0], partPoints[-1][0][1])
	#img = cv2.rectangle(img, start_point, end_point, (255, 255, 255), 5)
    
	input_pts = np.float32(partPoints) # PREPARE POINTS FOR WARP
	output_pts = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
	##############
	M = cv2.getPerspectiveTransform(input_pts, output_pts) # GET TRANSFORMATION MATRIX
	tmp1 = cv2.warpPerspective(img, M, (widthImg, heightImg), flags=cv2.INTER_LINEAR) # APPLY WARP PERSPECTIVE
	tmp1=tmp1[pad:tmp1.shape[0] - pad, pad:tmp1.shape[1] - pad]
	return tmp1


def cropSplitGetCells(img, partPoints,nr, nc, dim, pad):#widthImgg, heightImgg):

	
	start_point = (partPoints[0][0][0], partPoints[0][0][1])
	end_point = (partPoints[-1][0][0], partPoints[-1][0][1])
	img = cv2.rectangle(img, start_point, end_point, (255, 255, 255), pad)
	"""
	widthImg, heightImg = dim
	input_pts = np.float32(partPoints) # PREPARE POINTS FOR WARP
	output_pts = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
	##############
	M = cv2.getPerspectiveTransform(input_pts, output_pts) # GET TRANSFORMATION MATRIX
	tmp1 = cv2.warpPerspective(img, M, (widthImg, heightImg), flags=cv2.INTER_LINEAR) # APPLY WARP PERSPECTIVE
	"""

	widthImg, heightImg = dim
	tmp1 = wrap(img, partPoints, widthImg, heightImg, 0)

	tmpGray = cv2.cvtColor(tmp1,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
	tmp1Thresh = cv2.threshold(tmpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE
	if nr == 20 and nc == 6:
		tmp1Thresh = cv2.resize(tmp1Thresh,[192, 400],fx=0.5, fy=0.5)
	elif nc == 10:
		tmp1Thresh = cv2.resize(tmp1Thresh,[470, 155],fx=0.5, fy=0.5)		
	elif nc == 8:
		tmp1Thresh = cv2.resize(tmp1Thresh,[480, heightImg])#,fx=0.5, fy=0.5)
	boxes1 = splitBoxes(tmp1Thresh, nc, nr) # GET INDIVIDUAL BOXES
	return boxes1, tmp1#tmp1Thresh



def cannyFilter(img, gausFil, cannyFil, kernelParam, dilateParam, erodeParam):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
	imgBlur = cv2.GaussianBlur(imgGray, gausFil, 1) # ADD GAUSSIAN BLUR
	imgCanny = cv2.Canny(imgBlur,cannyFil[0],cannyFil[1]) # APPLY CANNY 
	kernel = np.ones(kernelParam)
	imgDial = cv2.dilate(imgCanny, kernel, iterations= dilateParam) # APPLY DILATION
	imgCanny = cv2.erode(imgDial, kernel, iterations= erodeParam)  # APPLY EROSION
	return imgCanny


def getresults(img1, students, widthImg, heightImg):
	#preprocessing
	img = preprocssing(img1, paperSize)
	showImages(["Orginal","Wraped"], 1, 2, [800, 800], img1, img)

	"""
	imgContours  = imgWarpColored.copy()

	imgCanny = cannyFilter(img, (5, 5), (10, 70), (1,1), 4, 4)
	contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS CHAIN_APPROX_NONE
	rectCon, imgContours = rectContour(contours, imgContours, 100) # FILTER FOR RECTANGLE CONTOURS
	cv2.drawContours(imgContours, rectCon, -1, (0, 0, 255), 2) # DRAW THE BIGGEST CONTOUR
	cv2.imshow("imgWarpColored", imgContours)
	cv2.waitKey(0)
	cornerPoints	= getCornerPoints(rectCon[-1]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
	cv2.drawContours(imgContours, cornerPoints, -1, (0, 255, 255), 2) # DRAW THE BIGGEST CONTOUR
	cv2.imshow("imgWarpColored", imgContours)
	cv2.waitKey(0)
	"""
	
	#img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
	imgFinal = img.copy()
	imgBlank = np.zeros((paperSize[1],paperSize[0], 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED

	imgCanny = cannyFilter(img, (5, 5), (10, 70), (1,1), 100, 100)
	

	## FIND ALL COUNTOURS
	imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
	imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES


	contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS CHAIN_APPROX_NONE	
	rectCon, imgContours = rectContour(contours, imgContours, 100) # FILTER FOR RECTANGLE CONTOURS

	ans1Points	= getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
	ans2Points	= getCornerPoints(rectCon[1]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
	ans3Points	= getCornerPoints(rectCon[2]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
	idPoints	= 	getCornerPoints(rectCon[3]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
	templatePoints = getCornerPoints(rectCon[4]) # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE
	gradePoints = getCornerPoints(rectCon[5]) # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE


	ans1Points, ans2Points, ans3Points = reorderContour(ans1Points, ans2Points, ans3Points)

	#try:
	if True:
		if ans1Points.size != 0 and ans2Points.size != 0 and ans3Points.size != 0 and idPoints.size != 0 and templatePoints.size != 0 and gradePoints.size != 0:
			c = (255,0,0)
			if 1:
				imgContours = drawRectangle(imgContours,ans1Points, c, 1)
				imgContours = drawRectangle(imgContours,ans2Points, c, 1)
				imgContours = drawRectangle(imgContours,ans3Points, c, 1)
				imgContours = drawRectangle(imgContours,idPoints, c, 1)
				imgContours = drawRectangle(imgContours,templatePoints, c, 1)
				imgContours = drawRectangle(imgContours,gradePoints, c, 1)
			else:		
				#"""
				cv2.drawContours(imgContours, ans1Points, -1, (0, 0, 255), 5) # DRAW THE BIGGEST CONTOUR
				cv2.drawContours(imgContours, ans2Points, -1, (0, 255, 255), 5) # DRAW THE BIGGEST CONTOUR
				cv2.drawContours(imgContours, ans3Points, -1, (0, 0, 255), 5) # DRAW THE BIGGEST CONTOUR
				cv2.drawContours(imgContours, idPoints, -1, (0, 255, 0), 5) # DRAW THE BIGGEST CONTOUR
				cv2.drawContours(imgContours, templatePoints, -1, (0, 255, 0), 5) # DRAW THE BIGGEST CONTOUR
				cv2.drawContours(imgContours, gradePoints, -1, (0, 255, 0), 5) # DRAW THE BIGGEST CONTOUR
				#"""


			ans1Points, dim1 =reorder(ans1Points) # REORDER FOR WARPING
			ans2Points, dim2 =reorder(ans2Points) # REORDER FOR WARPING
			ans3Points, dim23 =reorder(ans3Points) # REORDER FOR WARPING
			idPoints, dim3 =reorder(idPoints) # REORDER FOR WARPING
			templatePoints, dim4 =reorder(templatePoints) # REORDER FOR WARPING
			gradePoints, dim5 =reorder(gradePoints) # REORDER FOR WARPING
			
   

			ans1Boxes, i1     = cropSplitGetCells(img, ans1Points, 20, 6, dim1, 2)#widthImg, heightImg)
			ans2Boxes, i2     = cropSplitGetCells(img, ans2Points, 20, 6, dim2, 2)#widthImg, heightImg)
			ans3Boxes, i23     = cropSplitGetCells(img, ans3Points, 20, 6, dim23, 2)#widthImg, heightImg)
			idBoxes, i3       = cropSplitGetCells(img, idPoints, 5, 10, dim3, 3)#widthImg, heightImg)
			templateBoxes, i4 = cropSplitGetCells(img, templatePoints, 1, 8, dim4, 5)#widthImg, heightImg)#325, 25)
			gradeBoxes, i5    = cropSplitGetCells(img, gradePoints, 5, 10, dim5, 2)#widthImg, heightImg)
			
			showImages(["Orginal","Canny","contours",""], 2, 2, [paperSize[1]+100,paperSize[0]+100], imgFinal, imgCanny,imgContours, imgBlank)
			showImages(["Ans1-20","Ans21-40","Ans41-60","ID","Template","Grade"], 2, 3, [700, 700], i1, i2, i23, i3, i4, i5)
			
			idMarkedCells = getMarkedCells(idBoxes, 5, 10)
			temMarkedCells = getMarkedCells(templateBoxes, 1, 8)
			ans1MarkedCells = getMarkedCells(ans1Boxes, 20, 6)
			ans2MarkedCells = getMarkedCells(ans2Boxes, 20, 6)
			ans3MarkedCells = getMarkedCells(ans3Boxes, 20, 6)



			stdId = getMarked(idMarkedCells)
			print(stdId)
			stdTemplate = chr(65 + int(getMarked(temMarkedCells)))
			stdName = getNameById(students, int(stdId))

			print(stdId, "  ",stdName, "   ", stdTemplate)
			ans1 = getMarked(ans1MarkedCells)
			ans2 = getMarked(ans2MarkedCells)
			ans3 = getMarked(ans3MarkedCells)

			grading, rightAns = evaluateAns(ans1 + ans2 +ans3, stdName, stdTemplate)


			
			i111 = showAnswers(i1,rightAns[:20],grading[:20], ans1, ans1Points, dim1, 20, 6)
			i222 = showAnswers(i2,rightAns[20:40],grading[20:40], ans2, ans2Points, dim2, 20, 6)
			i333 = showAnswers(i23,rightAns[40:],grading[40:], ans3, ans3Points, dim23, 20, 6)
	

			imgFinal = maskPartAndReplace(imgFinal,ans1Points, i111, dim1)
			imgFinal = maskPartAndReplace(imgFinal,ans2Points, i222, dim2)
			imgFinal = maskPartAndReplace(imgFinal,ans3Points, i333, dim23)
			
			imgFinal, stdGrade = showScore(imgFinal, stdId, stdName, gradePoints, dim5, grading)
			showImages(["","","",""], 2, 2, [700, 700], i111, i222, i333, imgFinal)
			showImages([""],1, 1, [imgFinal.shape[1]+400,imgFinal.shape[0]+400], imgFinal)
	#except:
	else:
		pass

	return [stdId, stdName, np.ceil(stdGrade), stdTemplate], imgFinal, (stdTemplate,rightAns)

def showImages(lables,nr, nc, dimension, *imList):
	if not debug:
		return

	imageList = imList

	images = []
	row = []

	lablesNew = []
	rowLabel = []

	for r in range(0, len(imageList)):
		img = cv2.resize(imageList[r],dimension,fx=0.5, fy=0.5)
		#cv2.putText(img,lables[i], (20, 20),cv2.FONT_HERSHEY_PLAIN ,2,(255,255,0),2) # ADD THE GRADE TO NEW IMAGE
		row.append(img)
		rowLabel.append(lables[r])
		
		
		if (r+1) % nc == 0:
			images.append(row)
			lablesNew.append(rowLabel)
			row = []
			rowLabel = []
	stackedImage = stackImages(images,0.5,lablesNew)
	cv2.imshow("Result",stackedImage)
	cv2.waitKey(0)


def  reorderContour(ans1Points, ans2Points, ans3Points):
	ans = [ans1Points, ans2Points, ans3Points]
	myFunc = lambda x: x[0][0][0]
	ans.sort(key=myFunc)
	return ans


	
def maskPartAndReplace(img, imgPartPoints, imgPartial, dim):

	start_point = (imgPartPoints[0][0][0], imgPartPoints[0][0][1])
	end_point = (imgPartPoints[-1][0][0], imgPartPoints[-1][0][1])
	img = cv2.rectangle(img, start_point, end_point, (0, 0, 0), 3)
	#img = drawRectangle(img,imgPartPoints, (0,0,0), 8)

	ROI = imgPartPoints.copy()
	ROI[2] = ROI[3]
	ROI[3] = imgPartPoints[2]
	ROI = [
		[
			ROI[0][0], ROI[1][0], ROI[2][0], ROI[3][0]
		]
	]
	w = ROI[0][2]
	h = ROI[0][3]
	external_poly = np.array( ROI, dtype = np.int32 )
	cv2.fillPoly( img , external_poly, (0,0,0) )
	
	inn = np.float32(imgPartPoints) # PREPARE POINTS FOR WARP
	out = np.float32([[0, 0],[dim[0], 0], [0, dim[1]],[dim[0], dim[1]]]) # PREPARE POINTS FOR WARP
	#output_pts = np.float32([[0, 0],[dim[0], 0], [0, dim[1]],[dim[0], dim[1]]]) # PREPARE POINTS FOR WARP

	invMatrixG = cv2.getPerspectiveTransform(out, inn) # INVERSE TRANSFORMATION MATRIX	
	wrpedImgPartial = cv2.warpPerspective(imgPartial, invMatrixG, (img.shape[1], img.shape[0])) # INV IMAGE WARP



	img = cv2.addWeighted(wrpedImgPartial, 1, img, 1,1)
	return img

def showAnswers(imgPart,rightAns,grading, ans, ansPoints, dim, nr = 20, nc = 6):
	imgPart = drawGrid(imgPart,nr, nc)

	secH = int(imgPart.shape[0]/nr)
	secW = int(imgPart.shape[1]/nc)
	

	for x in range(0,nr):

		myAns= int(ans[x])
		cX = (myAns * secW) + secW // 2 
		cY = (x * secH) + secH // 2
		#print(cX, cY)
		
		if grading[x]==1:
			#cv2.rectangle(imgPart,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
			cv2.circle(imgPart,(cX,cY),10,(0,255,0),cv2.FILLED)
		else:

			#cv2.rectangle(imgPart, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
			
			# CORRECT ANSWER
			correctAns = int(rightAns[x])
			if myAns == 9:
				color = (255, 0, 0)
			else:
				color = (0, 255, 0)
				cv2.circle(imgPart,(cX,cY),10,(0,0,255),cv2.FILLED)

			#cv2.circle(imgPart, (cX, cY), 10, color, cv2.FILLED)

			cv2.circle(imgPart,((correctAns * secW)+secW//2, (x * secH)+secH//2),
			10,	color,cv2.FILLED)

	return imgPart



def drawGrid(img,nr,nc):
    secH = int(img.shape[0]/nr)
    secW = int(img.shape[1]/nc)
   

    for i in range (0, nc+2):
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt3, pt4, (0, 0, 0),1)

    for i in range (0, nr+2):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        cv2.line(img, pt1, pt2, (0, 0, 0),1)

    return img




def showScore(img, stdId, stdName, gradePoints, dim, grading):

	
	
	mist = 0
	doubleSolv = 0
	correct = 0

	for s in grading:
		if s == 0:
			mist = mist +1
		elif s == 9:
			doubleSolv = doubleSolv + 1
		else:
			correct = correct + 1
	
	score =  (correct/(len(grading)))*100 # FINAL GRADE

	x, y = gradePoints[0][0]
	#x, y = x + (dim[0]//2), y+ (dim[1]//2)
	x, y = x +50, y+25
	if score > 60:
		color = (0,255,0)
	else:
		color = (0,0,255)

	cv2.putText(img,str(np.ceil(score))+"%",(x-30, y-5)
                ,cv2.FONT_HERSHEY_PLAIN ,1,color,2) # ADD THE GRADE TO NEW IMAGE

	s = 5

	cv2.putText(img, stdId +"  " + stdName,(x-40, y-125+s), cv2.FONT_HERSHEY_PLAIN ,1,(0,0,0),2) # ADD THE GRADE TO NEW IMAGE
	cv2.putText(img, "correct" +"  " + str(correct),(x-40, y-100+s), cv2.FONT_HERSHEY_PLAIN ,1,(0,0,0),2) # ADD THE GRADE TO NEW IMAGE
	cv2.putText(img, "mistake" +"  " + str(mist),(x-40, y-75+s), cv2.FONT_HERSHEY_PLAIN ,1,(0,0,0),2) # ADD THE GRADE TO NEW IMAGE
	cv2.putText(img, "Error" +"  " + str(doubleSolv),(x-40, y-50+s), cv2.FONT_HERSHEY_PLAIN ,1,(0,0,0),2) # ADD THE GRADE TO NEW IMAGE


	return img, score

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]
    w = myPointsNew[3][0][0] - myPointsNew[0][0][0]
    h = myPointsNew[3][0][1] - myPointsNew[0][0][1]
    return myPointsNew, [w, h]


def rectContour(contours, imgContours, thr):

    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > thr:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
                #cv2.drawContours(imgContours, i, -1, (255, 0, 0), 4) # DRAW ALL DETECTED CONTOURS
	
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    return rectCon,imgContours



def biggestContour(contours, th, filter):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)

        if area > th:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == filter:
                biggest = approx
                max_area = area
    return biggest,max_area


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img, nr = 5, nc = 10):
    rows = np.vsplit(img, nc)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r, nr)
        for box in cols:
            boxes.append(box)
    return boxes




## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                if lables[d][c]!="":
                    #cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                    cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    return ver	

getNameById = lambda dataFramee, id :  dataFramee[dataFramee["ID"] == id]["name"].tolist()[0] 


def  getStudentDB(fileName):
	return pd.read_excel (fileName)

def getSolutions(solution_path, students, heightImg, widthImg):
	solutios_imgs = os.listdir(solution_path)
	solutiosNum = len(solutios_imgs)
	i = 0
	tempDict = {}
	while True:
		img = cv2.imread(solution_path + solutios_imgs[i]) 
		ress,imageSolution, here = getresults(img, students, heightImg, widthImg)
		tempDict[here[0]] = here[1]
		i = i + 1
		if i == solutiosNum:
			break
	return tempDict

def getTemAns(ans, stdName, stdTemplate):
	global solutios	
	if stdName == 'Teacher':
		solutios[stdTemplate] = ans
	return (solutios[stdTemplate])


def save2csv(fileName, header, rowData):
	# open the file in the write mode
	with open(fileName, 'w', encoding='UTF8') as f:
	    # create the csv writer
	    writer = csv.writer(f)
	    # write a row to the csv file
	    writer.writerow(rowData)


def save2excel(fileName, header, rowData):
	# Workbook is created
	wb = Workbook()
	# add_sheet is used to create sheet.
	sheet1 = wb.add_sheet('Sheet 1')

	row = 0

	for i, v in enumerate(header): 
		sheet1.write(row, i, header[i])

	for std in rowData:
		row = row + 1
		for i, v in enumerate(std): 
			sheet1.write(row,i, v)
	wb.save(fileName)

def init_data(studentsDBPath, solutionTemplates):
	global solutios
	students = getStudentDB(studentsDBPath)
	print("Students DATA BASE IMPORTED")
	print("Solutions DATA BASE IMPORTED")
	solutios = getSolutions(solutionTemplates, students, 700, 700)
	return students



def preprocssing(img, dimension):
    widthImg, heightImg = dimension
    #img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    #imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    
    #thres=valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    thres = (10, 70)

    imgThreshold = cannyFilter(img, (5, 5), thres, (1,1), 10, 10)
 
    # FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS


    #rectCon, imgContours = rectContour(contours, imgContours, 100000000000) # FILTER FOR RECTANGLE CONTOURS
    #cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2) # DRAW ALL DETECTED CONTOURS
    #cv2.imshow("Result",imgContours)
    #cv2.waitKey(0)
    # FIND THE BIGGEST COUNTOUR
   
    biggest, maxArea = biggestContour(contours, 5000, 4) # FIND THE BIGGEST CONTOUR
    #rectCon, imgContours = rectContour(contours, imgContours, 5000) # FILTER FOR RECTANGLE CONTOURS
    #biggest = rectCon[0]
    if biggest.size != 0:
        biggest, dim=reorder(biggest)
        paperSize = dim
        #cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 5) # DRAW THE BIGGEST CONTOUR
        #imgBigContour = drawRectangle(imgBigContour,biggest,2)
        imgWarpColored = wrap(img, biggest, dim[0], dim[1], 1)


        #REMOVE 20 PIXELS FORM EACH SIDE
        #imgWarpColored=imgWarpColored[0:imgWarpColored.shape[0] - 0, 0:imgWarpColored.shape[1] - 0]
        imgWarpColored = cv2.resize(imgWarpColored,dim)

        # APPLY ADAPTIVE THRESHOLD
        #imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        #imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        #imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        #imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        return imgWarpColored




def drawRectangle(img,biggest, color, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), color, thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[3][0][0], biggest[3][0][1]), color, thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), color, thickness)
    cv2.line(img, (biggest[1][0][0], biggest[1][0][1]), (biggest[2][0][0], biggest[2][0][1]), color, thickness)
 
    return img

"""
def nothing(x):
    pass
 
def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)
 
 
def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    #src = Threshold1,Threshold2
    src = 150, 160
    return src
"""