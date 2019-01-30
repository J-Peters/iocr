import cv2
import numpy as np
import re, sys, os, time, json
from collections import defaultdict

import PIL
from PIL import Image
import PIL.Image
from pytesseract import image_to_string
import pytesseract

from pdf2image import convert_from_path

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
#TESSDATA_PREFIX = 'C:/Program Files (x86)/Tesseract-OCR'

# import the necessary packages
from lenet import LeNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import keras
import numpy as np
import argparse
import cv2

federal_cls = {
                0 : 'Individual/sole proprietor or single-memeber LLC', 
		1 : 'C Corporation', 
		2 : 'S Corporation', 
		3 : 'Partenership',
		4 : 'Trust/estate', 
		5 : 'Limited liability company', 
		6 : 'Other'
		}


imSects = {
            '1' : {
	        'form_hdr_1'  : [0, 230, 0, 430],
                'form_hdr_2'  : [0, 230, 440,1900],
                'form_hdr_3'  : [0, 240, 2050,2400],
                'name'        : [240, 340, 155, 1800],
                'bus_nm'      : [340, 440, 180, 1800],
                'fed_tax_cls' : [440, 720, 170, 1910],
                'exemptions'  : [440, 720, 1905, 2500],
                'address'     : [720, 810, 160, 1600],
                'state'       : [820, 920, 160, 1600],
                'tin'         : [1000, 1420, 1700, 2500],
		'date'        : []
		  },
	    '2' : {
	        'form_hdr_1'  : [0, 230, 0, 430],
                'form_hdr_2'  : [0, 230, 440, 1900],
                'form_hdr_3'  : [0, 240, 2050, 2400],
                'name'        : [240, 340, 180, 1800],
                'bus_nm'      : [340, 440, 180, 1800],
                'fed_tax_cls' : [440, 850, 195, 1910],
                'exemptions'  : [440, 850, 1920, 2500],
                'address'     : [860, 945, 195, 1600],
                'state'       : [950, 1050, 160, 1600],
                'tin'         : [1180, 1550, 1700, 2500]
	      }
          }
w9_content = {}

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--pdf-file", type=str, default=-1, help="(mandatory) provide pdf file to be parsed")
ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
args = vars(ap.parse_args())
#print(args)

folder = os.path.join('static', 'files')
tmp_folder=os.path.join('static', 'tmp')

def get_pages(pdf_file):
  pages = convert_from_path(pdf_file, 300)
  head, tail = os.path.split(pdf_file)
  fileName = tail.split('.')[0]
  directory = os.path.join(folder, fileName)
  if not os.path.exists(directory):
     os.makedirs(directory)
  page_names = []
  for page in pages:
      imgfile = "{}-page_{}.jpg".format(fileName, pages.index(page))
      page.save(os.path.join(directory, imgfile), "JPEG")
      page_names.append(os.path.join(directory, imgfile))
  return page_names

def showim(img):
    winNm = "marked areas"; cv2.namedWindow(winNm); cv2.moveWindow(winNm, 20,20); cv2.imshow(winNm, img); cv2.waitKey(50000); cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
       reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
       i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b:b[1][i], reverse = reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def hlinesImg(arg, iter=1, vdilate=0):
    if type(arg) is np.ndarray: 
       img = arg.copy()
    else:
       img = cv2.imread(arg, 0)
    #img = cv2.imread(image_file, 0)
    (thresh, binImg) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binImg = 255 - binImg
    kernel_length = np.array(img).shape[1]//80
    vKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tmpImg = cv2.erode(binImg, hKernel, iterations=iter)
    hlinesImg = cv2.dilate(tmpImg, hKernel, iterations=1)
    (thresh, hlinesImg) = cv2.threshold(hlinesImg, 127,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if vdilate > 0: hlinesImg = cv2.dilate(tmpImg, vKernel, iterations=vdilate)
    return hlinesImg


def vlinesImg(arg):
    if type(arg) is np.ndarray: 
       img = arg.copy()
    else:
       img = cv2.imread(arg, 0)
    (thresh, binImg) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binImg = 255 - binImg
    kernel_length = np.array(img).shape[1]//80
    vKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tmpImg = cv2.erode(binImg, vKernel, iterations = 3)
    vlinesImg = cv2.dilate(tmpImg, vKernel, iterations = 3)
    (thresh, vlinesImg) = cv2.threshold(vlinesImg, 127,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return vlinesImg


def doc_angle(arg, folder=tmp_folder):
    if type(arg) is np.ndarray:
       image = arg.copy()
    else: 
       image = cv2.imread(arg)
    # need to pass only horizontal line images, if document has...
    if len(image.shape) > 2: image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges_doc = cv2.Canny(image,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges_doc,1,np.pi/180,200)
    return np.mean(np.array([x[0][1] for x in lines]))


def rotate_doc(image_file, filename='rotated.jpg', folder=tmp_folder):
    img = cv2.imread(image_file, 0)
    cv2.imwrite(os.path.join(folder, 'original.jpg'), img)
    (thresh, binImg) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hlineImg = hlinesImg(img)
    angle = np.rad2deg(doc_angle(hlineImg))-90
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binImg, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_img = rotated.copy()
    cv2.putText(rotated_img, 'Rotation Angle = '+str(angle), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(folder, filename), rotated_img)
    return rotated

def remove_noise(arg, filename, folder=tmp_folder):
    if type(arg) is np.ndarray:
       image = arg.copy()
    else: 
       image = cv2.imread(arg)
    retval, img = cv2.threshold(image, 100.0, 255.0, cv2.THRESH_BINARY)
    kern = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(img, kern, iterations=1)
    eroded = cv2.erode(dilated, kern, iterations=1)
    cv2.imwrite(os.path.join(folder, filename), eroded)
    return eroded

def crop_form(form, filename='cropped.jpg', folder=tmp_folder):
    img_orig = rotate_doc(form)
    (thresh, img_bin) = cv2.threshold(img_orig, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255-img_bin
    kernel = np.ones((5, 5), np.uint8)
    img_tmp = cv2.dilate(img_bin, kernel, iterations=20)
    im2, contours, hierarchy = cv2.findContours(img_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img_orig.copy()
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cropped_img = img_copy[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(folder, filename), cropped_img)
    return cropped_img


''' Tax payer identification number '''

def get_tin(img, filename='tin.jpg', folder=tmp_folder):
    (thresh, binImg) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binImg = 255 - binImg
    kernel_length = np.array(img).shape[1]//80;
    hKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    vKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hImg = cv2.erode(binImg, hKernel, iterations = 4)
    hlinesImg = cv2.dilate(hImg, hKernel, iterations = 1)
    vImg = cv2.erode(binImg, vKernel, iterations = 4)
    vlinesImg = cv2.dilate(vImg, vKernel, iterations = 1)
    finImg = cv2.add(hlinesImg, vlinesImg)
    dilate = cv2.dilate(finImg, hKernel, iterations = 2)
    vdilate = cv2.dilate(dilate, vKernel, iterations = 6)
    stripImg = cv2.bitwise_or(img, vdilate); #showim(stripImg)
    cv2.imwrite(os.path.join(folder, filename), stripImg)
    return cv2.resize(stripImg, (500, 500))

''' Federal tax classification '''
def get_fed_tax_cls(img):
    if len(img.shape) == 3:
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
       gray = img
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((1,8), np.uint8)
    #img_erosion = cv2.erode(thresh, kernel, iterations=1)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #find contours
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    #sort contours
    #sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    (sorted_ctrs, boundingBoxes) = sort_contours(ctrs, method="top-to-bottom")  
    idx = 0
    fed_tax_cls = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        if (w < 35 or h < 35) or (w > 100 or h > 100):
           continue
        #cv2.rectangle(img,(x,y),( x + w, y + h ),(0,0,255),2)
        roi = img[y:y+h, x:x+w]
        fed_tax_cls.append(np.sum(roi == 255))
        #showim(roi)
    return fed_tax_cls.index(min(fed_tax_cls))

def read_date(arg, weightsPath, numChannels=1, imgRows=28, imgCols=28, numClasses=10, filename='hwdates.jpg', folder=tmp_folder):
  outDt = ''; #showim(formDt); #print(formDt.shape)
  model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10, weightsPath=weightsPath)
  if type(arg) is np.ndarray:
     img = arg.copy()
     formDt = arg.copy()
  else: 
     img = cv2.imread(arg)
     formDt = cv2.imread(arg, 0)
  ret, thresh = cv2.threshold(~formDt, 127, 255, 0)
  image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  (sorted_ctrs, boundingBoxes) = sort_contours(contours, method="left-to-right")
  for i, c in enumerate(sorted_ctrs):
     tmp_img = np.zeros(formDt.shape, dtype=np.uint8)
     res = cv2.drawContours(tmp_img, [c], -1, 255, cv2.FILLED)
     tmp_img = np.bitwise_and(tmp_img, ~formDt)
     ret, inverted = cv2.threshold(tmp_img, 127, 255, cv2.THRESH_BINARY_INV)
     cnt = sorted_ctrs[i]
     x, y, w, h = cv2.boundingRect(cnt)
     cv2.rectangle(img,(x-1,y-1),( x + w + 1, y + h + 1),(0,255,0),2) 
     cropped = inverted[y:y + h, x:x + w]
     if (w < 15 and h < 15): continue
     cropped = cv2.bitwise_not(cropped)
     thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
     kernel = np.ones((2,2), np.uint8)
     gray_dilation = cv2.dilate(thresh, kernel, iterations=1)
     gray_erosion = cv2.erode(gray_dilation, kernel, iterations=1)
     gray_erosion=cv2.copyMakeBorder(gray_erosion, top=15, bottom=15, left=15, right=15, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
     the_img = cv2.resize(gray_erosion, (28, 28) )
     the_img = np.reshape(the_img, (1,28,28,1))
     probs = model.predict(the_img)
     prediction = probs.argmax(axis=1)
     outDt = outDt + str(prediction[0])
  cv2.imwrite(os.path.join(folder, filename), img)
  K.clear_session()
  return outDt[:2]+'-'+outDt[3:5]+'-'+outDt[6:]

if __name__ == "__main__":    
   start = time.time(); 
   pdf_file = args["pdf_file"]
   form = get_pages(pdf_file)[0]
   head, tail = os.path.split(pdf_file)
   fileName = tail.split('.')[0]
   if fileName == 'USB-0134388':
      frm_dt = [1960, 2050, 1700, 2500]
   elif fileName == 'USB-0134855':
      frm_dt = [2000, 2080, 1700, 2500]
   else:
      frm_dt = [2055, 2130, 1750, 2500]

   form_c = cv2.resize(crop_form(form), (2500, 3000))
   # get form header
   hdr_arr = imSects['1']['form_hdr_1']
   form_hdr = form_c[hdr_arr[0]:hdr_arr[1], hdr_arr[2]:hdr_arr[3]]
   cv2.imwrite(os.path.join(tmp_folder, 'form_hdr.jpg'), form_c[hdr_arr[0]:hdr_arr[1], hdr_arr[2]:hdr_arr[3]])
   year = re.findall(r"\D(\d{4})\D", pytesseract.image_to_string(Image.fromarray(form_hdr).convert("RGB"), lang='eng'))
   #print(year)
   if int(year[0]) < 2017:
      form_type = '1'
   else:
      form_type = '2'
   #print("FORM TYPE : ", form_type)

   # --------------------------------------------------------------------------------------------------------------------
   # capture field => name
   name_area = imSects[form_type]['name']
   name = form_c[name_area[0]:name_area[1], name_area[2]:name_area[3]]
   cv2.imwrite(os.path.join(tmp_folder, 'name.jpg'), name)
   name = pytesseract.image_to_string(Image.fromarray(name).convert("RGB"), lang='eng').splitlines(0)
   #print(name)
   if (len(name) > 1):
      w9_content['Name'] = name[-1]
   else:
      w9_content['Name'] = ''

   # --------------------------------------------------------------------------------------------------------------------
   # capture field => bus_name
   bus_name = imSects[form_type]['bus_nm']
   business_nm = form_c[bus_name[0]:bus_name[1], bus_name[2]:bus_name[3]]
   cv2.imwrite(os.path.join(tmp_folder, 'business_nm.jpg'), business_nm)    
   bus_nm = pytesseract.image_to_string(Image.fromarray(business_nm).convert("RGB"), lang='eng').splitlines(0)
   if (len(bus_nm) > 1):
      w9_content['Business Name'] = bus_nm[-1]
   else:
      w9_content['Business Name'] = ''
   # capture field => Federal tax classification
   fed_tax_area = imSects[form_type]['fed_tax_cls']
   w9_content['Federal Tax Classification'] = federal_cls[get_fed_tax_cls(form_c[fed_tax_area[0]:fed_tax_area[1], fed_tax_area[2]:fed_tax_area[3]])]

   # --------------------------------------------------------------------------------------------------------------------
   # capture field => exemptions
   exemp_area = imSects[form_type]['exemptions']
   exemptions = form_c[exemp_area[0]:exemp_area[1], exemp_area[2]:exemp_area[3]]
   cv2.imwrite(os.path.join(tmp_folder, 'exemptions.jpg'), exemptions)
   exemptions_arr = pytesseract.image_to_string(Image.fromarray(exemptions).convert("RGB"), lang='eng').splitlines(0)
   exemptions_idx = [exemptions_arr.index(y) for y in exemptions_arr if re.search('[\(|\{]if any[\)|\}$]', y)]
   payee_cd = exemptions_arr[exemptions_idx[0]]
   fatca = exemptions_arr[exemptions_idx[1]]
   w9_content['Exemptions - Payee Code'] = payee_cd[re.search('[\(|\{]*[\)|\}$]', payee_cd).start()+1:]
   w9_content['Exemptions - FATCA'] = fatca[re.search('[\(|\{]*[\)|\}$]', fatca).start()+1:]

   # --------------------------------------------------------------------------------------------------------------------
   # capture field => address
   add_area = imSects[form_type]['address']
   address = form_c[add_area[0]:add_area[1], add_area[2]:add_area[3]]
   cv2.imwrite(os.path.join(tmp_folder, 'address.jpg'), address)
   addr = pytesseract.image_to_string(Image.fromarray(address).convert("RGB"), lang='eng').splitlines(1)
   if (len(addr) > 1):
      w9_content['Address'] = addr[-1]
   else:
      w9_content['Address'] = ''

   # --------------------------------------------------------------------------------------------------------------------
   # capture field => city, area, zip_cd
   state_area = imSects[form_type]['state']
   state = pytesseract.image_to_string(Image.fromarray(form_c[state_area[0]:state_area[1], state_area[2]:state_area[3]]).convert("RGB"), lang='eng').splitlines(1)
   if len(state) > 1:
      w9_content['City, State, Zip'] = state[-1]
   else:
      w9_content['City, State, Zip'] = ''

   # --------------------------------------------------------------------------------------------------------------------
   # capture field => ssn/tin
   add_area = imSects[form_type]['tin']
   tin_area = form_c[add_area[0]:add_area[1], add_area[2]:add_area[3]]
   cv2.imwrite(os.path.join(tmp_folder, 'tin_area.jpg'), tin_area)
   idnum = pytesseract.image_to_string(Image.fromarray(get_tin(tin_area)).convert("RGB"), lang='eng'); #print('IDENTITY NUMBER => ', idnum)
   if re.match('[0-9]*\s*-[0-9]*\s*-[0-9]*\s*', idnum):
      w9_content['Social Security Number'] = idnum.replace(' ', '')
   else:
      w9_content['Tax Identification Number'] = idnum.replace(' ', '')
   
   # --------------------------------------------------------------------------------------------------------------------
   formDt = form_c[frm_dt[0]:frm_dt[1], frm_dt[2]:frm_dt[3]]; #showim(formDt)
   outDt = ''
   if fileName == 'USB-0134388':
      outDt = pytesseract.image_to_string(Image.fromarray(formDt).convert("RGB"), lang='eng'); 
   else:
      weightsPath = args['weights'] if args["load_model"] > 0 else None
      outDt = read_date(formDt, weightsPath)
   
   w9_content['dt'] = outDt[:2]+'-'+outDt[3:5]+'-'+outDt[6:]

   end = time.time()
   #print('Time elasped : ', end-start)
   w9_json = json.dumps(w9_content)
   print(json.dumps(json.loads(w9_json), indent=2, sort_keys=True))
