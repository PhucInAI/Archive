##########################################################################
# Get video data, will continue update infuture
# Author        :   Phuc Thanh Nguyen
# Created       :   Mar 13th, 2023
# Last edited   :   Mar 13th, 2023
##########################################################################


##########################################################################
# Import necessary libraries
##########################################################################


import  os
import  shutil
import  cv2
import  subprocess


##########################################################################
# Class of video
##########################################################################


class VideoInfo():
    """
        Class of video info
    """

    def __init__(self, video_path, delta_time   =   0.25):
        self.video_path     =   video_path
        self.vidieo_cap     =   cv2.VideoCapture(self.video_path)
        
        #-----------------------------------------------------------------
        # Time current parameter, use to get video by lenght
        #-----------------------------------------------------------------
        self.delta_time     =   delta_time      # Get 1 frame each delta_time (second)
        self.current_time   =   0
    
    #=====================================================================
    # Video metadata functions
    #=====================================================================
    def get_video_lenght(self):
        """
            Get video length with ffmpeg package
            Input   :   video path
            Output  :   lenght of video
        """
        
        result  =   subprocess.run([
                                        "ffprobe", 
                                        "-v", "error",
                                        "-show_entries",
                                        "format=duration", 
                                        "-of",
                                        "default=noprint_wrappers=1:nokey=1",
                                        self.video_path
                                    ],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
        
        return float(result.stdout)
    

    def get_video_fps(self):
        """
            Get video FPS
        """
        fps = self.vidieo_cap.get(cv2.CAP_PROP_FPS)

        return fps



    #=====================================================================
    # Video frames and others
    #=====================================================================


    def draw_change_by_time(
                                self,
                                score_lst,
                                patient_lst,
                                status_lst,
                                output_folder,
                                save_image = True,
                                save_video = True
                            ):
        """
            Draw video change by time
        """
        
        #-----------------------------------------------------------------
        # Make and clean output folder if it exists
        #-----------------------------------------------------------------
        output_path         =   os.path.join(output_folder, os.path.basename(self.video_path.split('.mp4')[0]))
        output_img_path     =   os.path.join(output_path, 'image')
        output_video_path   =   os.path.join(output_folder, os.path.basename(self.video_path.split('.mp4')[0]) + '.avi')
        

        if  os.path.exists(output_path):
            shutil          .rmtree(output_path)
        if  os.path.exists(output_video_path):
            shutil          .rmtree(output_video_path)
        
        os                  .mkdir(output_path)
        os                  .mkdir(output_img_path)

        #-----------------------------------------------------------------
        # Reset video time
        #-----------------------------------------------------------------
        self.current_time   =   0
        self.vidieo_cap     .set(cv2.CAP_PROP_POS_MSEC, self.current_time)
        
        #-----------------------------------------------------------------
        # Init parameters
        # current_draw  :   
        #   +   0, time of current frame < current_score time, pass away
        #   +   1, time neartest current_score time, draw
        #   +   2, time > current_score time, pass away
        #-----------------------------------------------------------------
        ret, frame          =   self.vidieo_cap.read()
        
        h, w, _             =   frame.shape
        video_fps           =   self.get_video_fps()
        video_writer        =   cv2.VideoWriter(
                                                    output_video_path,
                                                    cv2.VideoWriter_fourcc(*'DIVX'),
                                                    video_fps,
                                                    (w, h)
                                                )
        
        count               =   0   # frame number
        current_index       =   0   # index of current_score
        current_draw        =   0   # check if current frame is draw or not
        
        score_lastest       =   0
        patient_lastest     =   0
        status_lastest      =   ''

        while ret:
            #-------------------------------------------------------------
            # Get timestamp of frame
            #-------------------------------------------------------------
            time_in_frame   =   self.vidieo_cap.get(cv2.CAP_PROP_POS_MSEC)/1000

            #-------------------------------------------------------------
            #-------------------------------------------------------------
            # Get current_draw status and check update conditions
            if time_in_frame < self.current_time:
                current_draw    =   0
            
            elif time_in_frame >= self.current_time and current_draw == 0:
                #---------------------------------------------------------
                # Update lastest score, patient and status
                try:
                    score_lastest   =   score_lst[current_index]
                    patient_lastest =   patient_lst[current_index]
                    status_lastest  =   status_lst[current_index]
                except:
                    break
                
                #---------------------------------------------------------
                # Increase time and index
                current_index   +=  1
                self.current_time   += self.delta_time      # go to the next frame
                
                #---------------------------------------------------------
                # Update draw
                current_draw    =   1
            
            else:
                current_draw    =   2

            
            #---------------------------------------------------------
            # Put text to current frame
            #---------------------------------------------------------
            frame   =   cv2.putText(
                                        frame, 
                                        status_lastest,
                                        org= (50, 50),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,
                                        color=[255, 0, 0],
                                        thickness=2,
                                        lineType=cv2.LINE_AA
                                    )  
            frame   =   cv2.putText(    frame,
                                        'Score :{}'.format(str(score_lastest)),
                                        org= (50, 100),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,
                                        color=[255, 0, 0],
                                        thickness=2,
                                        lineType=cv2.LINE_AA
                                    ) 
            frame   =   cv2.putText(    frame, 
                                        'Patient: {}'.format(str(patient_lastest)),
                                        org= (50, 150),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1,
                                        color=[255, 0, 0],
                                        thickness=2,
                                        lineType=cv2.LINE_AA
                                    ) 

            #---------------------------------------------------------
            # Save frame to image and increase count
            #---------------------------------------------------------
            if current_draw ==  1:
                cv2.imwrite(os.path.join(output_img_path, 'frame_{}.jpg'.format(str(count).zfill(5))), frame)
                count  +=1

            #---------------------------------------------------------
            # Save frame to video
            #---------------------------------------------------------
            video_writer.write(frame)

            #---------------------------------------------------------
            # Move to next frame
            #---------------------------------------------------------
            ret, frame  =   self.vidieo_cap.read()


        video_writer.release()


    def get_frame_by_time(self):
        """
            Get frame of video by time
        """

        #-----------------------------------------------------------------
        # Set time and get frame
        #-----------------------------------------------------------------
        self.current_time   += self.delta_time                                          # calculate time
        self.vidieo_cap     .set(cv2.CAP_PROP_POS_MSEC, int(self.current_time*1000))    # set time
        ret, frame          =   self.vidieo_cap.read()                                  # get frame at time (ms)


        return  ret, frame