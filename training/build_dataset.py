import time
start_time = time.time()

from utils.iou import *
from constants import *
from utils.labels import *
from utils.selective_search import *
import cv2, os,logging

# loop over the output positive and negative directories
for dirPath in (POSITIVE_PATH, NEGATIVE_PATH):
	# if the output directory does not exist yet, create it
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

# initialize the total number of positive and negative images we have
# saved to disk so far
idx = 0
totalPositive = 0
totalNegative = 0

# loop over the image paths
for imagePath in SORTED_IMAGE_FILES:
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info("Processing image {}/{}...".format(idx + 1,len(SORTED_IMAGE_FILES)))
    
    image = cv2.imread(os.path.join(DIRECTORY_IMAGES,imagePath))
    getBoundingBoxes = extract_boxes(SORTED_LABEL_FILES[idx])
    idx +=1

    rectangles=selective_search(image)
    proposedRectangles= []
    for (x,y,width,height) in rectangles:
       proposedRectangles.append((x,y,x+width,y+height))
    # initialize counters used to count the number of positive and
    # negative ROIs saved thus far
    positiveROIs = 0
    negativeROIs = 0
    
    for proposedRectangle in proposedRectangles[:MAX_PROPOSALS]:
        propStartX, propStartY, propEndX, propEndY = proposedRectangle
        
        for trueBoundingBox in getBoundingBoxes:
            iou = intersection_over_union(trueBoundingBox, proposedRectangle)
            trueStartX, trueStartY, trueEndX, trueEndY = trueBoundingBox
            regionOfInterset = None
            outputPath = None
            
            if iou > 0.7 and positiveROIs <= MAX_POSITIVE:
                regionOfInterset = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalPositive)
                outputPath = os.path.sep.join([POSITIVE_PATH, filename])
                positiveROIs += 1
                totalPositive += 1
            
            fullOverlap = propStartX >= trueStartX
            fullOverlap = fullOverlap and propStartY >= trueStartY
            fullOverlap = fullOverlap and propEndX <= trueEndX
            fullOverlap = fullOverlap and propEndY <= trueEndY
            
            # check to see if there is not full overlap *and* the IoU
            # is less than 5% *and* we have not hit our negative
            # count limit
            if not fullOverlap and iou < 0.05 and negativeROIs <= MAX_NEGATIVE:
                regionOfInterset= image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalNegative)
                outputPath = os.path.sep.join([NEGATIVE_PATH, filename])
                negativeROIs += 1
                totalNegative += 1
                
            if regionOfInterset is not None and outputPath is not None:
                regionOfInterset = cv2.resize(regionOfInterset, INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, regionOfInterset)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info("Execution Time Of build_dataset.py : %s seconds " % (time.time() - start_time))
