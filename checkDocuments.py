#pip install openpyxl

import cv2
import numpy as np
import utilis
import os

heightImg = 510
widthImg  = 700

webCamFeed = False # AL edit later


#two modes( folder, video)
stdResults = {}
stdResults1= []

studentsDBPath = r""+os.getcwd() + "\students\students.xls"
solutionTemplates = os.getcwd() + "\\solutions\\"
students = utilis.init_data(studentsDBPath, solutionTemplates)

#utilis.initializeTrackbars()




if webCamFeed:
	cap = cv2.VideoCapture(0)
	cap.set(10,160)
	success, img = cap.read()
else:
	papers_path = os.getcwd() + "\\papers\\"
	papers_imgs = os.listdir(papers_path)
	papersNum = len(papers_imgs)
	
	i = 0
	while(i < papersNum):
		img = cv2.imread(papers_path + papers_imgs[i])
		
		#img_copy = img.copy()
		Result,img, _ = utilis.getresults(img, students, heightImg, widthImg)
		stdId, stdName, stdGrade, stdTemplate = Result
		stdResults[stdId] = [stdId, stdName, stdGrade, stdTemplate] 
		stdResults1.append([stdId, stdName, stdGrade, stdTemplate])
		cv2.imwrite(f"results/{stdId}_{stdName}_{papers_imgs[i]}",img)
		i = i + 1
	

	utilis.save2excel(os.getcwd() + "//results//res.xls",['ID', 'name', 'Grade','Template'],stdResults.values())
	print("Done....")


