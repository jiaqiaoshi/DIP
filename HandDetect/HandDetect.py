import cv2 as cv
import numpy as np

def detect_hand(img_hand):
    '''
    Detect hand from image
    Parameters:
    -----------
    img_hand: input image
    
    Return:
    -------
    hand_mask: results with 255 for hand pixel and 0 for background 
    '''
    hand_mask = np.zeros((img_hand.shape[0], img_hand.shape[1], 1), dtype = np.uint8)
    hand_mask=cv.cvtColor(img_hand,cv.COLOR_BGR2GRAY) # convert to grayscale
    hand_mask= cv.GaussianBlur(hand_mask,(5,5),0)
    ret,hand_mask= cv.threshold(hand_mask,700,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    hand_mask[:int(0.25*hand_mask.shape[0])]=0
    hand_mask[:,int(0.7*hand_mask.shape[1]):]=0
    hand_mask[int(0.9*hand_mask.shape[0]):,int(0.6*hand_mask.shape[1]):]=0
    return hand_mask

if __name__ == '__main__':
    video = cv.VideoCapture('hand.avi') 
   
    if (video.isOpened()== False): 
        print("Error opening video stream or file")
    # Read until video is completed
    i = 0
    while(video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Original Video',frame)
            # implement your detect_hand function
            hand_mask = detect_hand(frame)
            cv.imshow('Detection Results',hand_mask) 
            cv.imwrite('hand_mask_' + str(i) + '.bmp',hand_mask)
            i += 1   
                # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
# When everything done, release the video capture object
    video.release()

    # Closes all the frames
    cv.destroyAllWindows()
