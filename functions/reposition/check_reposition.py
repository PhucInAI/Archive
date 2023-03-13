##########################################################################
# Edge mapping to check whether a camera is stationary or moving
# Author        :   Phuc Thanh Nguyen
# Created       :   Mar 02nd, 2023
# Last edited   :   Mar 03nd, 2023
##########################################################################


##########################################################################
# Import necessary library
##########################################################################


import  cv2
import  numpy                               as      np
from    PIL                                 import  Image
from    tensorflow                          import  keras
from    keras.models                        import  load_model
from    functions.reposition.DenseDepth.layers  import BilinearUpSampling2D
from    functions.reposition.DenseDepth.utils   import predict
from    functions.ai_logging.ai_logging     import  logger


##########################################################################
# Class Camera Reposition
##########################################################################

class CameraReposition():
    """
        Class camera repositon
    """

    def __init__(self, map_rate=0.05, check_type='histogram', check_threshold=0.5, check_patient=3):
        """
            Input parameters:
                . map_rate          : map rate in rear, scalar on list of [top, left, right, bottom]
                . check_type        : algorithm to check, this version support Depthmap and Histogram
                . check_threshold   : threshold to classify whether there is a major change or not
                . check_patient     : threshold to assume there is a moving
            Other init parameters
                . previous_frame
                . current_frame
                . status
        """
        
        
        self.map_rate           =   self.get_map_rate(map_rate=map_rate)
        self.check_type         =   check_type
        self.check_threshold    =   check_threshold
        self.check_patient      =   check_patient
        self.current_patient    =   0
        self.previous_frame     =   None
        self.current_frame      =   None
        self.status             =   ''

        if self.check_type      ==  'depthmap':
            self.depthmap_model =   self.get_depthmap_model()
            self.previous_depthmap  =   None
            self.current_depthmap   =   None


    def get_map_rate(self, map_rate):
        """
            Get map_rate
        """


        if      type(map_rate)  ==  float:
            map_lst             =   [map_rate]*4

        elif    type(map_rate)  ==  list:
            map_lst             =   map_rate

        else:
            logger             .error('Type of map rate is wrong')

        return  map_lst
    

    def get_depthmap_model(self):
        """
            Load depthmap model
        """

        model_path          =   '../functions/reposition/DenseDepth/nyu.h5'
        custom_objects      =   {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
        model               =   load_model(model_path, custom_objects=custom_objects, compile=False)
        return  model
    

    def get_depthmap(self, frame):
        """
            Get depthmap from BGR frame
        """

        input_img           =   frame.copy()
        input_img           =   cv2.resize(input_img, (640, 480))
        input_img           =   cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img           =   np.clip(np.asarray(Image.fromarray(input_img), dtype=float) / 255, 0, 1)
        input_img           =   np.expand_dims(input_img, axis=0)
        depthmap            =   predict(self.depthmap_model, input_img)[0]*255
        depthmap            =   np.squeeze(depthmap, axis=-1)

        return  depthmap


    def get_different_score(self, frame):
        """
            Get different score from input image
        """
        
        #-----------------------------------------------------------------
        # If first frame
        #-----------------------------------------------------------------
        
        if  self.current_frame is None:
            self.current_frame      =   frame
            return  None, None, None
        

        #-----------------------------------------------------------------
        # Assign current frame
        #-----------------------------------------------------------------
        
        self.previous_frame         =   self.current_frame.copy()
        self.current_frame          =   frame


        #-----------------------------------------------------------------
        # Get difference score
        #-----------------------------------------------------------------
        
        if self.check_type          ==  'histogram':
            score                   =   self.score_difference_histogram()
        elif    self.check_type     ==  'depthmap':
            score                   =   self.score_difference_depthmap()

        if  score > self.check_threshold:
            self.current_patient    +=  1
            self.current_patient    =   min(self.current_patient, self.check_patient + 2)   # Assume that maximum state of patient = patient thresh + 2
        else:
            self.current_patient    =   max(self.current_patient-0.5, 0)                    # Avoid negative, current_patient >=0

        if  self.current_patient    >=  self.check_patient:
            self.status             =   'Reposition'
        else:
            self.status             =   'Stable'

        return  score, self.current_patient, self.status

    
    def score_difference_histogram(self):
        """
            Score different in edge of frame by histogram
        """
        
        #---------------------------------------------------------------------
        # Load grayscale and check size
        #---------------------------------------------------------------------

        previous_frame  =   cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        current_frame   =   cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        
        if not previous_frame.shape == current_frame.shape:
            logger.error('Size mismatch between 2 frames')


        #---------------------------------------------------------------------
        # Calculate euclician distance each part
        #---------------------------------------------------------------------
        
        total_distance  =   0
        h, w            =   previous_frame.shape
        
        for index, value in enumerate(self.map_rate):
            
            #-----------------------------------------------------------------
            # Get the part
            #-----------------------------------------------------------------
            if  index   ==  0:
                p_part  =   previous_frame[:, :int(value*w)]
                c_part  =   current_frame[:, :int(value*w)]
            elif index  ==  1:
                p_part  =   previous_frame[:int(value*h), :]
                c_part  =   current_frame[:int(value*h), :]
            elif index  ==  2:
                p_part  =   previous_frame[:, w-int(value*w):]
                c_part  =   current_frame[:, w-int(value*w):]
            else:
                p_part  =   previous_frame[h-int(value*h):, :]
                c_part  =   current_frame[h-int(value*h):, :]

            #-----------------------------------------------------------------
            # Calculate difference by histogram
            #-----------------------------------------------------------------
            p_histogram     =   cv2.calcHist([p_part], [0], None, [256], [0, 255])
            c_histogram     =   cv2.calcHist([c_part], [0], None, [256], [0, 255])
            distance        =   [(p_histogram[i]-c_histogram[i])**2 for i in range(min(len(p_histogram), len(c_histogram)))]
            distance        =   sum(distance)**(1/2)
            part_h, part_w  =   p_part.shape
            distance        =   distance/(part_h*part_w+1)
            
            total_distance  +=  distance

        return total_distance 


    def score_difference_depthmap(self):
        """
            Score different in edge of frame
        """

        #---------------------------------------------------------------------
        # Calculate depthmap
        #---------------------------------------------------------------------

        if self.current_depthmap is not None:       # if not first frame
            self.previous_depthmap  =   self.current_depthmap.copy()
        else:                                       # if first frame
            self.previous_depthmap  =   self.get_depthmap(self.previous_frame)

        self.current_depthmap       =   self.get_depthmap(self.current_frame)

        if not self.previous_depthmap.shape == self.current_depthmap.shape:
            logger.error('Size mismatch between 2 frames')

        #---------------------------------------------------------------------
        # Calculate euclician distance each part
        #---------------------------------------------------------------------
        total_distance  =   0
        h, w            =   self.previous_depthmap.shape
        for index, value in enumerate(self.map_rate):
            
            #-----------------------------------------------------------------
            # Get the part
            #-----------------------------------------------------------------
            if  index   ==  0:
                p_part  =   self.previous_depthmap[:, :int(value*w)]
                c_part  =   self.current_depthmap[:, :int(value*w)]
            elif index  ==  1:
                p_part  =   self.previous_depthmap[:int(value*h), :]
                c_part  =   self.current_depthmap[:int(value*h), :]
            elif index  ==  2:
                p_part  =   self.previous_depthmap[:, w-int(value*w):]
                c_part  =   self.current_depthmap[:, w-int(value*w):]
            else:
                p_part  =   self.previous_depthmap[h-int(value*h):, :]
                c_part  =   self.current_depthmap[h-int(value*h):, :]

            #-----------------------------------------------------------------
            # Calculate difference by histogram
            #-----------------------------------------------------------------
            p_histogram     =   cv2.calcHist([p_part], [0], None, [256], [0, 255])
            c_histogram     =   cv2.calcHist([c_part], [0], None, [256], [0, 255])
            distance        =   [(p_histogram[i]-c_histogram[i])**2 for i in range(min(len(p_histogram), len(c_histogram)))]
            distance        =   sum(distance)**(1/2)
            part_h, part_w  =   p_part.shape
            distance        =   distance/(part_h*part_w+1)
            
            total_distance  +=  distance

        return total_distance 