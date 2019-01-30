<strong><h2>Intelligent OCR</h2></strong>
The utility processes scanned pdf forms. The program needs to know the structure of the document to extract chunks of document and feed to opnCV and pytesseract to extract information (user entries) ##

The following command triggers the server on `port 8080`
```
python iocr.py
```

`fucntions` supported:
- [x] @read_handwritten_dates
- [x] @crop_region_of_interest
- [x] @remove noise
- [x] @rotate_images
- [x] @read_textboxes
- [x] @mark_checkboxes
