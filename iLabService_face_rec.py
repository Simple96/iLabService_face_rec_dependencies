from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./demos/')
#from classifier import getRep
#from classifier import getRep_by_image
import os
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import cv2
from imgaug import augmenters as iaa
import numpy as np
np.set_printoptions(precision=4)
import datetime 
import pickle
import openface
import time
#import classifier models from sklearn
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#import mtcnn dependencies
import tensorflow as tf
import detect_face
import dlib
#extra
import pandas as pd
from operator import itemgetter

ROOT_DIR = "/home/ilabservice/openface"
DATA_DIR = "/home/ilabservice/openface/face_data"
CLASSIFIER_DIR = "/home/ilabservice/openface/generated-embeddings"
LABEL_FILE_DIR = "/home/ilabservice/openface"
ALIGNED_IMAGE_DIR = "/home/ilabservice/openface/face_data1"#"/home/ilabservice/openface/aligned-images"
TEST_PIC_DIR = "/home/ilabservice/openface/real_time_pic"
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DLIB_MODEL_DIR = os.path.join(MODEL_DIR, 'dlib')
OPENFACE_MODEL_DIR = os.path.join(MODEL_DIR, 'openface')

Image_dim = 96#every 'face' is of size 96*96 
Data_augmented = False
Data_ready = False
Raw_Data_Num = 5
Cuda_Flag = False

def get_all_bounding_boxes_mtcnn(frame):
	'''
	use MTCNN network to acquire aligned bounding boxes
	input: numpy array, a single frame(image)
	output: norf_faces 	- int, detected face number
			det 		- 2d numpy array(norf_faces+1 rows, 4 columns), detected faces
						  4 columns represents:
						  0: left, 1: top, 2: right, 3: bottom
	'''
	minsize = 20  # minimum size of face
	threshold = [0.6, 0.7, 0.7]  # three steps's threshold
	factor = 0.709  # scale factor
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, '/det')

	start = time.time()
	bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
	print("time interval = ",format(time.time() - start))

	nrof_faces = bounding_boxes.shape[0]
	print('Detected_FaceNum: %d' % nrof_faces)

	if nrof_faces > 0:
		det = bounding_boxes[:, 0:4]
		img_size = np.asarray(frame.shape)[0:2]

	return nrof_faces, det

def get_largest_bounding_box_mtcnn(frame, skipMulti = False):
	'''
	use MTCNN network to acquire the largest aligned bounding boxes
	input: numpy array, a single frame(image)
	output: dlib.rectangle(), the data structure is {[left, top], [right, bottom]}
	'''
	assert frame is not None
	
        faces, det = get_all_bounding_boxes_mtcnn(frame)
	rect = []
	#print("instance type is:",type(det[0][0]))
	for instance in det:
		rect.append(dlib.rectangle(left = int(instance[0]),top=int(instance[1]),right=int(instance[2]),bottom=int(instance[3])))
        if (not skipMulti and len(rect) > 0) or len(rect) == 1:
            return max(rect, key=lambda rect: rect.width() * rect.height())
        else:
            return None
	
	return None
    

def getRep_by_image(bgrImg, multiple=False, dlib = True):
	'''
	get vector representation of a single image
	input: a single bgr image(set to a 96*96 face)
	output: a 128-dim vvector(numpy array)
	'''
    dlibModel_path = os.path.join(DLIB_MODEL_DIR,"shape_predictor_68_face_landmarks.dat")
    align = openface.AlignDlib(dlibModel_path)
    network_model = os.path.join(OPENFACE_MODEL_DIR,'nn4.small2.v1.t7')
    net = openface.TorchNeuralNet(network_model, imgDim=Image_dim, cuda=Cuda_Flag)
    
        #dlibModel_path = os.path.join(DLIB_MODEL_DIR,"shape_predictor_68_face_landmarks.dat")
        #align = openface.AlignDlib(dlibModel_path)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    start = time.time()
    if dlib:
        if multiple:
            bbs = align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = align.getLargestFaceBoundingBox(rgbImg)
	    print("bb1 = ",bb1)
	    print("bb1 type is:",type(bb1))
            bbs = [bb1]
    else:
        bbs = []
        for i in range(nrof_faces):
	    bbs.append(dlib.rectangle(det[i][0], det[i][1], det[i][2], det[i][3]))
        
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face")
    print("bbs = ", bbs)
    print("aligning faces take {} minutes", time.time() - start)

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            Image_dim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))

        start = time.time()
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def real_time_face_rec_service():
	'''
	for future development
	'''
	while True:
		os.system("")
	return




def add_user_(img, user_name):
	'''
	input:  img should be a list of length 5
			each element in the list should be an image - numpy 2d array
			user_name, string, the name of the input images
	'''
	save_path = DATA_DIR+"/"+user_name+"/"
	print("save path is:", save_path)
	if img is None or user_name is None:
		logger.error("Wrong input of add_user_(). Please check.")
		return False
	if not os.path.isdir(save_path):
		os.system("mkdir " + save_path)
	if len(img) != Raw_Data_Num:
		logger.warning("Wrong picture number")
	for i in range(len(img)):
		cv2.imwrite(save_path+user_name+str(i)+".jpg",img[i])	
	data_aug(DATA_DIR+"/"+user_name)
	acquired_aligned_images(DATA+"/"+user_name, ALIGNED_IMAGE_DIR+"/"+user_name)
	extract_features()
	#TODO: add user in the database	
	return True

def delete_user_(user_name):
	'''
	delete the user images files
	'''
	name_list = os.listdir(DATA_DIR)
	for i in range(len(name_list)):
		if name_list[i] == user_name:
			#TODO: delete user in the database
			os.system("rm -r " + DATA_DIR + "/" + username)
			
	name_list = os.listdir(ALIGNED_IMAGE_DIR)
        for i in range(len(name_list)):
                if name_list[i] == user_name:
                        #TODO: delete user in the database
                        os.system("rm -r " + ALIGNED_IMAGE_DIR + "/" + username)

	logger.warning("can't find the user you wanna delete : < ")
	return False

def list_all_files(rootdir):
	'''
	sub-funciton of data_aug
	'''
	_files = []
	list = os.listdir(rootdir) #list all files and dirs under the file
	for i in range(0,len(list)):
		path = os.path.join(rootdir,list[i])
		if os.path.isdir(path):
                	_files.extend(list_all_files(path))
		if os.path.isfile(path):
			_files.append(path)
	return _files

def data_aug(data_path):
	'''
	augment data
	increase data number
	generate 10 extra pictures from 1 picture 
	This function defines 13 different augment methods
	Everytime would choose 2 randomly and use the combination of these 2 methods to process all the images under the input data_path
	the processed data would still be under the original data directory
	'''
	list = list_all_files(data_path)#os.listdir(data_path) 
	for i in range(0,len(list)):
                #path = os.path.join(data_path,list[i])
		path =  list[i]

                #if os.path.isfile(path):
		try:
			img = cv2.imread(path)
			print("read path succeed: ",path)
			#print("image shape is: ", img.shape)
		except:
			print("Image read error. Please check the path again!")

		else:
                        #11 different kinds of pre-processing operators
                        #
			q1 = iaa.Alpha((0.0, 1.0),first=iaa.MedianBlur(9),per_channel=True)
			#alpha noise
			q2 = iaa.SimplexNoiseAlpha(first=iaa.EdgeDetect(0.5),per_channel=False)
			#noise in the frequency domain
			q3 = iaa.FrequencyNoiseAlpha(first=iaa.Affine(rotate=(-10, 10),translate_px={"x": (-4, 4), "y": (-4, 4)}),second=iaa.AddToHueAndSaturation((-40, 40)),per_channel=0.5)
			#set 5% of all the pixels black
			q4 = iaa.Dropout(p=0.05, per_channel=False, name=None, deterministic=False, random_state=None)
			#adjust contrast to make the image darker
			q5 = iaa.ContrastNormalization(alpha=1.5, per_channel=False, name=None, deterministic=False, random_state=None)
			#adjust contrast to make the image brighter
			q6 = iaa.ContrastNormalization(alpha=0.5, per_channel=False, name=None, deterministic=False, random_state=None)
			#16 pixels left
			q7 = iaa.Affine(translate_px={"x": -16})
			#sharpen
			q8 = iaa.Sharpen(alpha=0.15, lightness=1, name=None, deterministic=False, random_state=None)
			#emboss, like sharpen
			q9 = iaa.Emboss(alpha=1, strength=1, name=None, deterministic=False, random_state=None)
			#fliplr, upside down
			q10 = iaa.Fliplr(1.0)
			#gaussian blur
			q11 = iaa.GaussianBlur(3.0)
			#scale y axis randomly x0.8-1.2 
			q12 = iaa.Affine(scale={"y": (0.8, 1.2)})
			#scale x axis randomly x0.8-1.2
			q13 = iaa.Affine(scale={"x": (0.8, 1.2)})
			#randomly combine 2 of all the operations
			q = iaa.SomeOf(2,[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13])
                        
                        #save_path1 = os.path.dirname(path) + "/aug1_" + path.split('/')[-1].split('.')[0] + ".jpg"
                        #print("save_path1 is : ", save_path1)
                        #save pre-processed images
			for i in range(10):
				#augment each image by 10 randomly chosen methods  
				img_aug = q.augment_images([img])
				print("img_aug type is:", type(img_aug))
				#generate save path
				save_path = os.path.dirname(path) + "/aug"+str(i)+"_" + path.split('/')[-1].split('.')[0] + ".jpg"
				#save images
				cv2.imwrite(save_path,img_aug[0])

def acquired_aligned_images(data_dir, aligned_image_dir):
	os.system(ROOT_DIR + "/util/align-dlib.py " + data_dir + "/" + " align outerEyesAndNose " + aligned_image_dir + "/ --size 96")

def extract_features():
	'''
	acquire feature vectors of all the existed data
	a necessary step before training
	the results are two csv files named labels.csv and reps.csv and would be put into CLASSIFIER_DIR
	'''
	os.system(ROOT_DIR + "/batch-represent/main.lua -outDir " + CLASSIFIER_DIR + "/ -data " + ALIGNED_IMAGE_DIR )

def training(method = 0):
	'''
	just training the SVM model by existed data
	the result would be put into CLASSIFIER_DIR
	'''
	training_methods_dic = {0:'LinearSvm',1:'GridSearchSvm',2:'GMM',3:'RadialSvm',4:'DecisionTree',5:'GaussianNB',6:'DBN'}
	os.system(ROOT_DIR + "/demos/classifier.py train " + "--classifier=" + training_methods_dic[method] + " " + CLASSIFIER_DIR)

def classify(img):
	'''
	given an image, give you the class it belongs
	input:numpy array, a single image
	output: name - the name of the class it belongs to
			distance - the distance between the vector of input image and the average vector of the class it belongs to
			confidence - the confidence number indicating how much likely the image belongs to the class
						 this is calculate by a build-in funciton of sklearn
	you can either dicede weather the classificaiton is good by confidence or distance
	'''
	if not isinstance(img, np.ndarray):
		logger.error("wrong input of classifier: input variable is not an image!")
		return -1
		print('Loadign classifier ...')		
	#This part could be optimized
	classifier_path = CLASSIFIER_DIR + "/classifier.pkl"
	with open(classifier_path, 'rb') as f:
        	if sys.version_info[0] < 3:
                	(le, clf) = pickle.load(f)
        	else:
                	(le, clf) = pickle.load(f, encoding='latin1')
	#This part could be optimized 
	#Like load in the main funciton
	'''
	print('Loading feature info ...')
        model_info = {}
        with open("./feature_info.json",'r') as json_file:
            model_info = json.load(json_file)
        array_model_info = np.array(model_info)

	print('Loading label info ...')
        with open("./labels_info.json",'r') as load_f:
            labels_info = json.load(load_f)
	'''

	print('Loading succeed, start recognition.')
	start = time.time()

	#This part could be optimized 
	print("Loading embedding info...")
	fname = "{}/reps.csv".format(CLASSIFIER_DIR)
	embeddings = pd.read_csv(fname, header=None).as_matrix()
	fname = "{}/labels.csv".format(CLASSIFIER_DIR)
    	labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    	labels = map(itemgetter(1),
                 	map(os.path.split,
                     		map(os.path.dirname, labels)))  # Get the directory.
	lel = LabelEncoder().fit(labels)
	labelsNum = lel.transform(labels)
	print("embedding info loaded succeed.")

	reps = getRep_by_image(img)
	print("representations took {} seconds"),format(time.time()-start)
	if len(reps) > 1:
        	print("List of faces in image from left to right")
	for r in reps:
        	rep = r[1].reshape(1, -1)
        	bbx = r[0]
        	start = time.time()
        	predictions = clf.predict_proba(rep).ravel()
        	print("predictions are:")
        	print(predictions)
        	maxI = np.argmax(predictions)
        	person = le.inverse_transform(maxI)
		print("person type is:",type(str(person)))

		person_embedding = np.zeros(embeddings.shape[1])
		embedding_count = 0
		for i in range(len(labels)):
			if str(person) == labels[i]:
				person_embedding = person_embedding + embeddings[i]
				embedding_count = embedding_count + 1
		person_embedding = person_embedding / embedding_count
		print("person_embedding is:", person_embedding)
		print("distance is:", abs(np.sum(rep - person_embedding)))
		confidence = predictions[maxI]
        print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
            
         if isinstance(clf, GMM):
			dist = np.linalg.norm(rep - clf.means_[maxI])
			print("  + Distance from the mean: {}",format(dist))
	return person.decode('utf-8'), abs(np.sum(rep - person_embedding)), confidence

def cal_threshold():
	'''
	calculate the distance threshold
	'''
	namelist = os.listdir(DATA_DIR)
	lowerboundlist = []
	for name in namelist:
		path = DATA_DIR + "/" + name
		filename = os.listdir(path)[0]
		img_path = path + "/" + filename
		img = cv2.imread(img_path)
		name_cla, distance = classify(img)
		print(name + " is classifiered as: " + name_cla)
		lowerboundlist.append(distance)
	lowerbound = np.mean(lowerboundlist)
	img2 = cv2.imread("my_test_cases/test_pic_han.jpg")
	name_cla, upperbound = classify(img)
	return 1.0*(lowerbound + upperbound)/ 2.0


#align = openface.AlignDlib(args.dlibFacePredictor)		
if __name__ == '__main__':
	#log
	logger.info("Start print log")
	logger.debug("Do something")
	#load dlib and get align class
	#this class is used for acquiring aligned data
	dlibModel_path = os.path.join(DLIB_MODEL_DIR,"shape_predictor_68_face_landmarks.dat")
	align = openface.AlignDlib(dlibModel_path)
	#load network model and get net class
	#this class is used for get embedding representation - 128 dim vectors
	network_model = os.path.join(OPENFACE_MODEL_DIR,'nn4.small2.v1.t7')
	net = openface.TorchNeuralNet(network_model, imgDim=Image_dim, cuda=Cuda_Flag)
	#load classifier
	#the trained SVM classifier to classify mulltiple 128-dim vectors
	classifier_path = CLASSIFIER_DIR + "/classifier.pkl"

    #test image
    #for testing, you can modify your path below
    img = cv2.imread("my_test_cases/test_pic_hailong.jpg")
	time_start = time.time()
	name, distance, confidence = classify(img)
	if distance > 0.7 or confidence < 0.3:
		print("sorry, the classificaiton result may not be right.")
	time_interval = time.time() - time_start
	logger.info("training takes"+str(time_interval) + " seconds")
	logger.info("Finish")	

'''
def init_status_():
	Training_flag = False
	Data_augmented = False

	list = os.listdir(DATA_DIR) #list all files and dirs under the file
	#for i in range(0,len(list)):
        #	path = os.path.join(rootdir,list[i])
        #	if os.path.isdir(path):
        #        	_files.extend(list_all_files(path))
        #    	if os.path.isfile(path):
        #        	_files.append(path)
	person_num = len(list)
	Data_ready = False
	return

	#img_list = []
	#load_dir = "/home/ilabservice/openface/temp_dir/linkang/"
	#for i in range(5):
	#	temp_path = load_dir + "linkang" + str(i+1) + ".jpg"
	#	print(temp_path)
	#	img_list.append(cv2.imread(temp_path))
	#add_user_(img_list, "linkang")
	#data_aug(DATA_DIR+"/linkang")
	#acquired_aligned_images()
	#extract_features()


test code:

main
	with open(classifier_path, 'rb') as f:
                if sys.version_info[0] < 3:
                        (le, clf) = pickle.load(f)
                else:
                        (le, clf) = pickle.load(f, encoding='latin1')

	#name, distance = classify(img)
	#print("a single predict takes {} seconds",format(time.time() - time_start))
	#acquired_aligned_images("/home/ilabservice/openface/face_data/000", "/home/ilabservice/openface/test_dir")
	#training(0)
	#norf_faces, det = get_all_bounding_boxes_mtcnn(img)
	#det = align.getLargestFaceBoundingBox(img)
	#print("norf_face is:")
	#print(norf_faces)
	#det = dlib.rectangle(left=1, top=1, right=2, bottom=2) 
	#det= get_largest_bounding_box_mtcnn(img)
	#print("det is:")
	#print(det)
	#print("det type is:",type(det))
	#cal_threshold()
	#extract_features()

classify
#t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	#test_im_save_path = TEST_PIC_DIR + "/" + t + ".jpg"
	#save_status = cv2.imwrite(test_im_save_path,img)
	#if not save_status:
	#	logger.error("saving image failed in classifier, can't classifier the image!")
	#	return save_status
	#os.system(ROOT_DIR + "/demos/classifier.py infer " + CLASSIFIER_DIR + "/classifier.pkl " + test_im_save_path)

    	'''
