from    logging                         import raiseExceptions
import  threading
import  gi
import  sys
import  packages
import  os
import  time

packages.include()
if 'gi' in sys.modules:
    gi.require_version('Gst',       '1.0')
    gi.require_version('GstBase',   '1.0')
    gi.require_version('GstVideo',  '1.0')
    from    gi.repository                   import  Gst, GLib, GObject, GstVideo  # noqa:F401,F402

from system.vms_simulation                  import  VmsSimulation
from ai_functions.base_system.ai_processor  import  AiProcessor

import  torch
from    multiprocessing                     import  Process
import  camera.listcam                      as      listcam


#=========================================================================
# Global - Define cameras in use
#=========================================================================
cameras                     =   [listcam.deen_cam3]

#=========================================================================
# This function is used to split the system into isolated
# processes with the same dudy
#=========================================================================


def split_processes(cameras,nb_cameras_each_process):
    """
        Split cameras into group of processes
        Input:
            . cameras                   :   List of camera
            . nb_cameras_each_process   :   Max number of camera each processes
        Output
            . processes                 :   List of processes object in system
    """

    #---------------------------------------------------------------------
    # Split cameras into groups
    #---------------------------------------------------------------------
    
    camera_groups           =   []
    each_group              =   []
    
    for count, camera in enumerate(cameras):
        
        each_group          .append(camera)
        
        if (count+1)%nb_cameras_each_process == 0:
            camera_groups   .append(each_group)
            each_group      =   []
            

    #---------------------------------------------------------------------
    # Split camera groups into processes
    #---------------------------------------------------------------------
    
    processes = []
    
    for each_camera_group in camera_groups:        
        #-----------------------------------------------------------------
        # Create processes for ai and append
        #-----------------------------------------------------------------
        process             =   Process(target=run_system, args=(each_camera_group))
        processes           .append(process)
    

    return processes

   
#=========================================================================
# This function is used to run the Ai in each process independently
#=========================================================================


def run_system(cameras):

    vms_simulation          =   VmsSimulation()
    
    
    #---------------------------------------------------------------------
    # Init the ai system
    #---------------------------------------------------------------------
    
    ai_processor            =   AiProcessor("./data", "./ai_core/model/" ,isTensorrt=True)
    
    if ai_processor.gpu_is_available:

        ai_processor        .start()
        vms_simulation      .start()
        ai_processor        .add_camera(
                                        vms_simulation.init_cameras,
                                        vms_simulation.init_in_buffer,
                                        vms_simulation.init_in_condition,
                                        vms_simulation.init_out_buffer,
                                        vms_simulation.init_out_condition
                                        )
    else:
        raiseExceptions("GPU is not available")

    
#=========================================================================
# MAIN FUCNTION 
#=========================================================================


if __name__ == '__main__':
    
    nb_cameras_each_process =   len(cameras)
    Gst.init(None)
    main_loop               =   GLib.MainLoop()
    
    processes               =   split_processes(cameras,nb_cameras_each_process)

    #---------------------------------------------------------------------
    # Start the processes
    #---------------------------------------------------------------------
    
    print('start process', len(processes))
    for process in processes:
        print('start process 1')
        process             .start()
    

    try:
        main_loop           .run()
    except:
        main_loop           .quit()
    
    
    #---------------------------------------------------------------------
    # Wait untill all the child process complete
    #---------------------------------------------------------------------
    
    for proc in processes:
        proc                .join()   
