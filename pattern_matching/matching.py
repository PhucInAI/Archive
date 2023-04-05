import  cv2
import  numpy               as  np 
import  matplotlib.pyplot   as  plt

def template_match(img, template_path):    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[1], template.shape[0]

    # Apply threshold to find matching pattern 
    res = cv2.matchTemplate(img_gray,template, cv2.TM_CCOEFF_NORMED)
    THRESHOLD = 0.5
    loc = np.where(res >= THRESHOLD)

    if len(loc[0])>0:
        return loc[1][0], loc[0][0], w, h   # x, y, w, h
    else:
        return None