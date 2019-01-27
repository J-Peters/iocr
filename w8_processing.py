import base_functions as adv
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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--pdf-file", type=str, default=-1, help="(mandatory) provide pdf file to be parsed")
ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
args = vars(ap.parse_args())
#print(args)

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files (x86)/Tesseract-OCR'

imSects = {
           'pg1' : {
			  'form_hdr_1'  : [0, 100, 0, 200],
              'form_hdr_2'  : [0, 240, 460, 2000],
              'form_hdr_3'  : [0, 230, 2020, 2400],
              'name'        : [760, 860, 0, 1500],
			  'country'     : [760, 860, 1500, 2400],
              'disr_ent'    : [860, 920, 0, 2400],
              'ch_3_sts_a'  : [930, 1100, 0, 2500],
			  'ch_3_sts_b'  : [1100, 1200, 0, 2500],
              'ch_4_sts'    : [1200, 2380, 0, 2500],
              'perm_addr'   : [2380, 2490, 0, 2500],
              'perm_city'   : [2480, 2580, 0, 1200],
              'perm_ctry'   : [2480, 2580, 1850, 2400],
              'mail_addr'   : [2580, 2660, 0, 2500],
			  'mail_city'   : [2670, 2750, 0, 1850],
              'mail_ctry'   : [2670, 2750, 1850, 2400],
              'us_tin'      : [2760, 2860, 0, 700],
			  'giin'        : [2760, 2860, 950, 1350],
			  'f_tin'       : [2760, 2860, 1350, 1780],
			  'ref_nbr'     : [2760, 2860, 1800, 2500]
	      },
		  'pg2' : {
			  'part2'       : [190, 760, 0, 2500],
              'part3'       : [760, 1350, 0, 2500],
              'part4'       : [1420, 2170, 0, 2500],
			  'part5'       : [2240, 2890, 0, 2500]
		  }
		 }

ch_3_sts = {
            '1'  : 'Corporation', 
			'2'  : 'Disregarded entity',
            '3'  : 'Partnership',
			'4'  : 'Simpie trust',
            '5'  : 'Grantor trust',
            '6'  : 'Complex trust',
            '7'  : 'Estate',
            '8'	 : 'Government',
			'9'  : 'Central Bank of Issue',
            '10' : 'Tax-exempt organization',
            '11' : 'Private foundation',
			'12' : 'Yes',
            '13' : 'No'
		}
ch_4_sts = {
 '1' : 'Nonparticipating FFI (including a limited FFI or an FFI related to. Reporting IGA FFI other than a registered deemed-compliant FF or participating FFI).',
 '2' : 'Participating FFI.',
 '3' : 'Reporting Model 1 FFI.',
 '4' : 'Reporting Model 2 FFI.',
 '5' : 'Registered deemed-compliant FFI (other than a reporting Model FFI or sponsored FFI that has not obtained a GIIN).',
 '6' : 'Sponsored FFI that has not obtained a GIIN. Complete Part IV.',
 '7' : 'Certified deemed-compiiant nonregistering local bank. Compltet Part V.',
 '8' : 'Certified deemed-compliant FFS with only low-value accounts. Complete Part VI.',
 '9' : 'Certified deemed-compliant sponsored, closely held investmen: vehicle. Complete Part VII.',
 '10' : 'Certified deemed-compliant limited life debt investment entity. Complete Part VIII',
 '11' : 'Certitied deemed-campliant investment advisors and investment managers. Complete Part IX.',
 '12' : 'Owner-documented FFI. Complete Part X.',
 '13' : 'Restricted distributor. Complete Part XI.',
 '14' : 'Nonreporting IGA FFI (including an FF! treated as a registered deemed-compliant FFI under an applicable Model 2 IGA). Complete Part XIl.',
 '15' : 'Foreign government, government of a U.S. possession, or foreign central bank of issue. Complete Part XIII.',
 '16' : 'International organization. Complete Part XIV.',
 '17' : 'Exempt retirement plans. Complete Part XV.',
 '18' : 'Entity wholly owned by exempt beneficial owners. Complete Part XVI.',
 '19' : 'Territory financial institution. Complete Part XVII.',
 '20' : 'Nonfinancial group entity. Complete Part XVIII.',
 '21' : 'Excepted nonfinancial start-up company. Complete Part XIX.',
 '22' : 'Excepted nonfinancial entity in liquidation or bankruptcy. Complete Part XX.',
 '23' : '501(c) organization. Compiete Part XXI.',
 '24' : 'Nonprofit organization. Complete Part XXII.',
 '25' : 'Publicly traded NFFE or NFFE affiliate of a publicly traded corporation. Complete Part XXIlE.',
 '26' : 'Excepted territory NFFE. Complete Part XXIV.',
 '27' : 'Active NFFE. Complete Part XXVV.',
 '28' : 'Passive NFFE. Complete Part XXVI.',
 '29' : 'Excepted inter-affiliate FFI. Complete Part XXVII.',
 '30' : 'Direct reporting NFFE.',
 '31' : 'Sponsored direct reporting NFFE. Compiete Part XXVIII.'
}


# variables - folders and imgae files
w8_content = {}
for i in range(30):
    w8_content['part'+('0'+str(i+1) if i < 9 else str(i+1))] = {}

binOpt = {'1' : "Yes", '2' : "No"}
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

def get_area(img, pg_num, sect_nm):
    return img[imSects[pg_num][sect_nm][0]:imSects[pg_num][sect_nm][1], imSects[pg_num][sect_nm][2]:imSects[pg_num][sect_nm][3]]

def checkbox(img, filename, txtWidth=0, count=1, meth="top-to-bottom", folder=tmp_folder):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if len(img.shape) == 3:
       gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
       gray = img
    
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    im2,ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours
    (sorted_ctrs, boundingBoxes) = adv.sort_contours(ctrs, method=meth)
    fresh = 255 - thresh
    chkbx_arr = []; obj = {}
    for c in sorted_ctrs:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.15 * peri, True)
        # if  approximated contour has four points, then
        # it is assumed that it is a checkboxes
        x, y, w, h = cv2.boundingRect(c)
        if len(approx) == 4 and w > 35:
           roi = fresh[y:y+h, x:x+w]
           #adv.showim(roi_desc)
           sum = np.sum(cv2.resize(roi, (50, 50)) == 255)
           if txtWidth > 0:
              roi_desc = pytesseract.image_to_string(Image.fromarray(fresh[y:y+h, x+w:x+w+txtWidth]).convert("RGB"), lang='eng')
              obj[roi_desc] = sum
              cv2.rectangle(img,(x,y),( x + w + txtWidth, y + h ),(0,0,255),2)
              cv2.putText(img, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
           else:
              cv2.rectangle(img,(x,y),( x + w, y + h ),(0,0,255),2)
              cv2.putText(img, str(count), (x-w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
              chkbx_arr.append(sum)
           #adv.showim(img)
           count += 1
    filename = filename if len(filename.split('.')) > 1 else filename+'.jpg'
    cv2.imwrite(os.path.join(folder, filename), img)
    return obj if txtWidth > 0 else chkbx_arr


def get_line_vals(arg, filename, folder=tmp_folder, count=1):
    if type(arg) is np.ndarray: 
       img = arg.copy()
    else:
       img = cv2.imread(arg, 0)
    dimg = hlinesImg(img, iter=10, vdilate=1); #adv.showim(dimg); 
    cv2.imwrite(os.path.join(folder, filename), dimg)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #if len(dimg.shape) == 3:
    #   gray = cv2.cvtColor(dimg,cv2.COLOR_BGR2GRAY)
    #else:
    #   gray = dimg
    #ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    im2, ctrs, hier = cv2.findContours(dimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours
    (sorted_ctrs, boundingBoxes) = adv.sort_contours(ctrs, method="top-to-bottom")
    lines_arr = []
    for c in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(c)
        output = pytesseract.image_to_string(Image.fromarray(img[y+h//2-50:y+h//2, x:x+w]).convert("RGB"), lang='eng')
        cv2.rectangle(img,(x,y+h//2-50),( x + w, y + h//2 ),(0,0,255),2);#adv.showim(img)
        cv2.putText(img, str(count), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        count += 1
        lines_arr.append(output)
    cv2.imwrite(os.path.join(folder, filename), img)
    return lines_arr

def pg_scan(page, pg_name, idx):
    obj = checkbox(page, pg_name)
    for key in range(len(obj)):
        if obj[key] < 1300:
           w8_content[idx[key].split(':')[0]][idx[key].split(':')[1]] = 'Yes'


if __name__ == "__main__":    
   start = time.time()
   pdf_file = args["pdf_file"]
   page_names = get_pages(pdf_file)
   page_1 = page_names[0]
   page_2 = page_names[1]
   page_3 = page_names[2]
   page_4 = page_names[3]
   page_5 = page_names[4]
   page_6 = page_names[5]
   page_7 = page_names[6]
   page_8 = page_names[7]
   # crop first page and set standard size of (width = 2500, height = 3000)
   form_c = cv2.resize(adv.crop_form(page_1), (2500, 3000))
   
   #----------------------------------
   '''     P  A  G  E   -   1      '''
   #----------------------------------
   
   # ----------------------------------------------------------------------
   # ------------ PAGE 1 - PART I  -  1, 2, 3  ----------------------------
   # ----------------------------------------------------------------------
   name = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'name')).convert("RGB"), lang='eng').splitlines()
   country = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'country')).convert("RGB"), lang='eng').splitlines()
   disr_ent = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'disr_ent')).convert("RGB"), lang='eng').splitlines()
   
   w8_content['part01']['01'] = name[-1] if len(name) > 1 else ''
   w8_content['part01']['02'] = country[-1] if len(country) > 1 else ''
   w8_content['part01']['03'] = disr_ent[-1] if len(disr_ent) > 1 else ''
   
   ## ----------------------------------------------------------------------
   ## ------------  PAGE 1 - PART I - 4 (Chapter 3 Status)  ----------------
   ## ----------------------------------------------------------------------
   
   #checkbox(img, filename, txtWidth=0, count=1, folder='D:\\DB_OCR\\tmp')
   w8_content['part01']['04a'] = []
   
   obj_a = checkbox(get_area(form_c, 'pg1', 'ch_3_sts_a'), 'ch-3-sts-a', txtWidth=400)
   for key in obj_a.keys():
       if obj_a[key] < 1300:
          w8_content['part01']['04a'].append(key)

   obj_b = checkbox(get_area(form_c, 'pg1', 'ch_3_sts_b'), 'ch-3-sts-b', meth="left-to-right")
   for i in range(len(obj_b)):
       if obj_b[i] < 1300: w8_content['part01']['04b'] = binOpt[str(i+1)]
   
   ## ----------------------------------------------------------------------
   ## ------------  PAGE 1 - PART I - 5 (Chapter 4 Status)  ----------------
   ## ----------------------------------------------------------------------
   img = get_area(form_c, 'pg1', 'ch_4_sts')
   w8_content['part01']['05'] = []
   
   h, w = img.shape
   obj_a = checkbox(img[:, 0:w//2], 'ch-4-sts-a')
   obj_b = checkbox(img[:, w//2:w], 'ch-4-sts-b', count=len(obj_a)+1)
   ch4_arr = [obj_a.index(x) for x in obj_a if x < 1300] + [obj_a.index(y) for y in obj_b if y < 1300]
   w8_content['part01']['05'] = [ch_4_sts[str(key)] for key in ch4_arr]
   
   ## ----------------------------------------------------------------------
   ## ------------  PAGE 1 - PART I - 6, 7, 8, 9a, 9b, 10   ----------------
   ## ----------------------------------------------------------------------
   perm_addr = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'perm_addr')).convert("RGB"), lang='eng').splitlines()
   perm_city = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'perm_city')).convert("RGB"), lang='eng').splitlines()
   perm_ctry = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'perm_ctry')).convert("RGB"), lang='eng').splitlines()
   mail_addr = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'mail_addr')).convert("RGB"), lang='eng').splitlines()
   mail_city = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'mail_city')).convert("RGB"), lang='eng').splitlines()
   mail_ctry = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'mail_ctry')).convert("RGB"), lang='eng').splitlines()
   us_tin    = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'us_tin')).convert("RGB"), lang='eng').splitlines()
   giin      = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'giin')).convert("RGB"), lang='eng').splitlines()
   f_tin     = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'f_tin')).convert("RGB"), lang='eng').splitlines()
   ref_nbr   = pytesseract.image_to_string(Image.fromarray(get_area(form_c, 'pg1', 'ref_nbr')).convert("RGB"), lang='eng').splitlines()
   
   w8_content['part01']['06']  = perm_addr[-1] if len(perm_addr) > 1 else ''
   w8_content['part01']['06a'] = perm_city[-1] if len(perm_city) > 1 else ''
   w8_content['part01']['06b'] = perm_ctry[-1] if len(perm_ctry) > 1 else ''
   w8_content['part01']['07']  = mail_addr[-1] if len(mail_addr) > 1 else ''
   w8_content['part01']['07a'] = mail_city[-1] if len(mail_city) > 1 else ''
   w8_content['part01']['07b'] = mail_ctry[-1] if len(mail_ctry) > 1 else ''
   w8_content['part01']['08']  = us_tin[-1]    if len(us_tin)    > 1 else ''
   w8_content['part01']['09a'] = giin[-1]      if len(giin)      > 1 else ''
   w8_content['part01']['09b'] = f_tin[-1]     if len(f_tin)     > 1 else ''
   w8_content['part01']['10'] = ref_nbr[-1]    if len(ref_nbr)   > 1 else ''
   
   
   # crop second page and set standard size of (width = 2500, height = 3000)
   form_c = cv2.resize(adv.crop_form(page_2), (2500, 3000))
   #----------------------------------
   '''     P  A  G  E   -   2      '''
   #----------------------------------
   
   img = get_area(form_c, 'pg2', 'part2')
   w8_content['part02']['11'] = []
   obj = checkbox(img, 'pg2-part2', txtWidth=400)
   for key in obj.keys():
       if obj[key] < 1300:
          w8_content['part02']['11'].append(key)
   
   
   img = get_area(form_c, 'pg2', 'part3')
   w8_content['part02']['14'] = []
   obj = checkbox(img, 'pg2-part3')
   for key in range(len(obj)):
       if obj[key] < 1300:
          w8_content['part03']['11'].append(key)
   
   
   img = get_area(form_c, 'pg2', 'part4')
   w8_content['part02']['17'] = []
   obj = checkbox(img, 'pg2-part4')
   for key in range(len(obj)):
       if obj[key] < 1300:
          w8_content['part04']['17'] = 'Yes'
   
   
   img = get_area(form_c, 'pg2', 'part5')
   obj = checkbox(img, 'pg2-part5')
   for key in range(len(obj)):
       if obj[key] < 1300:
          w8_content['part05']['18'] = 'Yes'
   

   #------------------------------------------
   '''     P  A  G  E  S -   3,4,5,6,7     '''
   #------------------------------------------

   p3_idx = {'part06:19', 'part07:21', 'part08:22', 'part09:23', 'part10:24a'}
   pg_scan(cv2.resize(adv.crop_form(page_3), (2500, 3000)), 'pg3', p3_idx)
   
   p4_idx = ['part10:24b', 'part10:24c', 'part10:24d', 'part11:25a', 'part11:25b', 'part11:25c', 'part12:26']
   pg_scan(cv2.resize(adv.crop_form(page_4), (2500, 3000)), 'pg4', p4_idx)
   
   p5_idx = ['part13:27', 'part14:28a', 'part14:28b', 'part15:29a', 'part15:29b', 'part15:29c', 'part15:29d', 'part15:29e']
   pg_scan(cv2.resize(adv.crop_form(page_5), (2500, 3000)), 'pg5', p5_idx)
   
   p6_idx = ['part15:29f', 'part16:30', 'part17:31', 'part18:32', 'part19:33', 'part20:34', 'part21:35']
   pg_scan(cv2.resize(adv.crop_form(page_6), (2500, 3000)), 'pg6', p6_idx)
   
   p7_idx = ['part22:36', 'part23:37a', 'part23:37b', 'part24:38', 'part25:39', 'part26:40a', 'part26:40b', 'part26:40c', 'part27:41']
   pg_scan(cv2.resize(adv.crop_form(page_7), (2500, 3000)), 'pg7', p7_idx)

   #----------------------------------
   '''     P  A  G  E   -   8      '''
   #----------------------------------
   
   model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10, weightsPath=args["weights"] if args["load_model"] > 0 else None)
   #model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10, weightsPath="D:\\DB_OCR\\output\\lenet_weights.hdf5")

   img_8 = cv2.imread(page_8)
   gray = cv2.cvtColor(img_8,cv2.COLOR_BGR2GRAY)
   (thresh, binImg) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   binImg = 255 - binImg

   kernel = np.array((2,2), np.uint8)
   binImg = cv2.erode(binImg, kernel, iterations = 2)
   binImg = cv2.dilate(binImg, kernel, iterations = 1)

   binImg = 255 - binImg
   cv2.imwrite('{}\\pg8_pre.jpg'.format(tmp_folder), binImg)

   page_8_mod = '{}\\pg8_pre.jpg'.format(tmp_folder)
   form_c = cv2.resize(adv.crop_form(page_8_mod), (2500, 3000))

   p8_idx = ['part28:43', 'part29:sign']
   pg_scan(form_c, 'pg8', p8_idx)

   outDt = ''
   formDt = form_c[1200:1280, 2100:2500]; #adv.showim(formDt) #formDt = form_c[1100:1210, 2050:2500]
   ret, thresh = cv2.threshold(~formDt, 127, 255, 0)
   image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   (sorted_ctrs, boundingBoxes) = adv.sort_contours(contours, method="left-to-right")
   for i, c in enumerate(sorted_ctrs):
       tmp_img = np.zeros(formDt.shape, dtype=np.uint8)
       res = cv2.drawContours(tmp_img, [c], -1, 255, cv2.FILLED)
       tmp_img = np.bitwise_and(tmp_img, ~formDt)
       ret, inverted = cv2.threshold(tmp_img, 127, 255, cv2.THRESH_BINARY_INV)
       cnt = sorted_ctrs[i]
       x, y, w, h = cv2.boundingRect(cnt)
       cropped = inverted[y:y + h, x:x + w]
       cropped_orig = formDt[y:y + h, x:x + w]
       if (w < 15 and h < 15): continue
       #adv.showim(cv2.resize(cropped, (100, 100)))
       cropped = cv2.bitwise_not(cropped)
       thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
       kernel = np.ones((2,2), np.uint8)
       gray_dilation = cv2.dilate(thresh, kernel, iterations=1)
       gray_erosion = cv2.erode(gray_dilation, kernel, iterations=1)
       gray_erosion = cv2.copyMakeBorder(gray_erosion, top=15, bottom=15, left=15, right=15, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
       #print(gray_erosion.shape)
       the_img = cv2.resize(gray_erosion, (28, 28))
       the_img = np.reshape(the_img, (1,28,28,1))
       #cv2.putText(cropped_orig, str(prediction[0]), (2, 2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
       #adv.showim(cv2.resize(gray_erosion, (100, 100)))
       probs = model.predict(the_img)
       prediction = probs.argmax(axis=1)
       #print(prediction[0])
       outDt = outDt + str(prediction[0])
   
   w8_content['part29']['dt'] = outDt[:2]+'-'+outDt[3:5]+'-'+outDt[6:]
   
   
   print(json.dumps(w8_content, sort_keys=True,indent=4, separators=(',', ': ')))
