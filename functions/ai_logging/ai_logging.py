##########################################################################
# Simple logging config, will update "more" beautiful in future
# Author        :   Phuc Nguyen Thanh
# Created       :   Mar 08th, 2023
# Last editted  :   Mar 08th, 2023
##########################################################################

import  sys
import  logging

#-------------------------------------------------------------------------
# Basic logger
#-------------------------------------------------------------------------
logger  =   logging.getLogger('ai_logger')
logging .basicConfig(
                        level       =   logging.INFO,
                        format      =   "%(asctime)s [%(levelname)s] %(message)s",
                        handlers    =   [
                                            logging.FileHandler("debug.log"),
                                            logging.StreamHandler(sys.stdout)
                                        ]
                    )