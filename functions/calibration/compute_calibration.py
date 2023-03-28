##########################################################################
# Compute calibration (projection) matrix from 2 gievn images
# Author        :   Phuc Thanh Nguyen
# Created       :   Mar 28th, 2023
# Last edited   :   Mar 28th, 2023
# Use DeepPTZ to compute camera calibration matrix
##########################################################################


##########################################################################
# Import libraries
##########################################################################


import  cv2
import  numpy                                           as      np
from    skimage                                         import  io, transform

import  torch
from    torchvision                                     import  transforms

from    inception                                       import  MyInception3_siamese
from    inception_efficient                             import  MyInception3_siamese_efficient


##########################################################################
# Model
##########################################################################





##########################################################################
# Post-processing
##########################################################################





##########################################################################
# ComputeCalibration class
##########################################################################


class CameraCalibration():
    """
        Class Camera Calibration
    """
    
    
    def __init__(
                    self,
                    devide,
                    efficient   =   True,
                    checkpoint  =   './pretrained/checkpoint_efficient.pth.tar',
                    focal_ratio =   10,
                    mean_focal  =   275,
                ):
        
        self.device             =   devide
        self.model              =   self.init_model(efficient, checkpoint)

        normalize               =    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform          =   transforms.Compose([transforms.ToTensor(), normalize, ])
    
        self.fr                 =   focal_ratio
        self.mean_focal         =   mean_focal


    def init_model(self, efficient, checkpoint):
        """
            Build model and load checkpoint
            Input:
                . efficient     :   check if use efficient network ot not
                . checkpoint    :   weights path
            Output
                . model         :   model use to predict
        """
        
        
        #-----------------------------------------------------------------
        # Build model
        #-----------------------------------------------------------------
        
        if efficient:
            model               =   MyInception3_siamese_efficient  (
                                                                    siamese             =   True,
                                                                    pretrained          =   False,
                                                                    pretrained_fixed    =   False,
                                                                    aux_logits          =   False
                                                                    )
        else:
            model               =   MyInception3_siamese(           
                                                                    siamese             =   True,
                                                                    pretrained          =   False,
                                                                    pretrained_fixed    =   False,
                                                                    aux_logits          =   False
                                                        )
        

        #-----------------------------------------------------------------
        # Load weights
        #-----------------------------------------------------------------
        network_data            =   torch.load(checkpoint)
        model                   .load_state_dict(network_data['state_dict'])
        model                   =   torch.nn.DataParallel(model).to(self.device)
        model                   .eval()
        print('Total Parameters: %.3fM' %(sum(p.numel() for p in model.parameters())/1000000.0))
        
        
        return model


    def load_image(self, img_path):
        """
            Load image (torch Tensor in device) from image path
        """
        # img                     =   io.imread(img_path)
        img                     =   cv2.imread(img_path)
        
        new_shape               =   (299,299)
        original_shape          =   (img.shape[1], img.shape[0])
        ratio                   =   float(max(new_shape))/max(original_shape)
        new_size                =   tuple([int(x*ratio) for x in original_shape])
        img                     =   cv2.resize(img, new_size)
        delta_w                 =   new_shape[0] - new_size[0]
        delta_h                 =   new_shape[1] - new_size[1]
        top, bottom             =   delta_h//2, delta_h-(delta_h//2)
        left, right             =   delta_w//2, delta_w-(delta_w//2)
        img                     =   cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        img                     =   cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img                     =   img/255.0
        img                     =   self.transform(img)
        img                     =   torch.unsqueeze(img, 0)
        # img                     =   torch.from_numpy(img)
        img                     =   img.float().to(self.device)
        return  img


    def post_processing(self, roll, pitch, yaw, focal1, focal2, dist1, dist2):
        """
            Reference : https://stackoverflow.com/questions/21412169/creating-a-rotation-matrix-with-pitch-yaw-roll-using-eigen/21412445#21412445
        """
        
        #-----------------------------------------------------------------
        # Intrinsic Matrix
        #-----------------------------------------------------------------
        in_matrix               =   np.array(
                                                [
                                                    [focal1,      0, dist1, 0],
                                                    [0     , focal2, dist2, 0],
                                                    [0     ,      0,     1, 0]
                                                ]
                                            )
        

        #-----------------------------------------------------------------
        # Pitch, Yaw, Roll Matrix
        #-----------------------------------------------------------------
        pitch_matrix            =   np.array(
                                                [
                                                    [1,              0,             0, 0],
                                                    [0,  np.cos(pitch), np.sin(pitch), 0],
                                                    [0, -np.sin(pitch), np.cos(pitch), 0],
                                                    [0,              0,             0, 1]
                                                ]
                                            )
        yaw_matrix              =   np.array(
                                                [
                                                    [np.cos(yaw), 0, -np.sin(yaw), 0],
                                                    [          0, 1,            0, 0],
                                                    [np.sin(yaw), 0,  np.cos(yaw), 0],
                                                    [          0, 0,            0, 1]
                                                ]
                                            )
        roll_matrix             =   np.array(
                                                [
                                                    [ np.cos(roll), np.sin(roll), 0, 0],
                                                    [-np.sin(roll), np.cos(roll), 0, 0],
                                                    [            0,            0, 1, 0],
                                                    [            0,            0, 0, 1]
                                                ]
                                            )
        

        #-----------------------------------------------------------------
        # Extrinsic Matrix
        #-----------------------------------------------------------------
        ex_matrix               =   np.dot(np.dot(pitch_matrix, yaw_matrix), roll_matrix)


        #-----------------------------------------------------------------
        # Projection matrix
        #-----------------------------------------------------------------
        projection_matrix       =   np.dot(in_matrix, ex_matrix)

        
        return in_matrix, ex_matrix, projection_matrix


    def compute_with_image(self, img_1_path, img_2_path):
        img_1                   =   self.load_image(img_1_path)
        img_2                   =   self.load_image(img_2_path)
       
        output                  =   self.model(img_1, img_2).cpu().detach().numpy()[0]
        
        roll1                   =   output[0]
        pitch1                  =   output[1]
        yaw1                    =   output[2]
        focal11                 =   output[3]*self.fr+self.mean_focal
        focal12                 =   output[4]*self.fr+self.mean_focal
        dist11                  =   output[5]/10 + 0.5
        dist12                  =   output[6]/10 + 0.5
        roll2                   =   output[7]
        pitch2                  =   output[8]
        yaw2                    =   output[9]
        focal22                 =   output[10]*self.fr+self.mean_focal
        focal21                 =   output[11]*self.fr+self.mean_focal
        dist22                  =   output[12]/10 + 0.5
        dist21                  =   output[13]/10 + 0.5

        _, _, projection_matrix =   self.post_processing(roll2, pitch2, yaw2, focal21, focal22, dist21, dist22)
        
        print(projection_matrix)

        return projection_matrix
    

##########################################################################
# Main, for testing only
##########################################################################


def main():
    
    img_1_path                  =   '/home/ptn/Storage/viAct/viAct/functions/calibration/data/test/image_00/data/0000000000.png'
    img_2_path                  =   '/home/ptn/Storage/viAct/viAct/functions/calibration/data/test/image_00/data/0000000011.png'
    
    devide                      =   'cuda'
    calibration_obj             =   CameraCalibration(devide=devide)

    calibration_obj             .compute_with_image(img_1_path, img_2_path)



if __name__                     ==  '__main__':
    main()