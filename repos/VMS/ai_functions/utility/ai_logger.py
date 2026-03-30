#===========================================
# This function is used to define the log file 
# format. To get more detail about this log, 
# please reference the following sites: 
# https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial
# https://docs.python.org/3/library/logging.config.html#logging-config-dictschema
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
# https://stackoverflow.com/questions/13805905/python-logging-dictconfig
#
#
# To use this file, import the logger file by:
#   from ai_core.utility.ai_logger import aiLogger
#   
#   aiLogger.debug("your message");     -> use for tesing temporarily
#   aiLogger.info("your message");      -> use for showing the info that is rarely modified 
#   aiLogger.exception("your message"); -> place this one inside an exception
#   aiLogger.error("your message")      -> using when having some error
#   aiLogger.warning("your message")    -> using when having some warning
# 
# Notice: The "your message" is similar to the way you use the print command
#
# Author: An Nguyen
# Created day: July 7, 2022
# Last modify: Sep 5, 2022
#===========================================


import logging.config
import os


class CustomFormatter(logging.Formatter):
    
    black           = "\x1b[30m"

    red             = "\x1b[31m"
    boldRed         = "\x1b[31;1m"
    italicsRed      = "\x1b[31;3m"

    green           = "\x1b[32m"
    boldGreen       = "\x1b[32;1m"
    italicsGreen    = "\x1b[32;3m"
    lightGreen      = "\x1b[92m"
    boldLightGreen = "\x1b[92;1m"

    yellow          = "\x1b[33m"
    boldYellow      = "\x1b[33;1m"
    italicsYellow   = "\x1b[33;3m"

    blue            = "\x1b[34m"
    boldBlue        = "\x1b[34;1m"
    italicsBlue     = "\x1b[34;3m"


    lightBlue       = "\x1b[94"
    boldLightBlue   = "\x1b[94;1;3m"


    magenta         = "\x1b[35m"            #dark purple
    boldMagenta     = "\x1b[95;1m" 
    italicsMagenta  = "\x1b[35m;m" 

    
    cyan            = "\x1b[36m"            #dark green
    boldCyan        = "\x1b[36;1m"
    italicsCyan     = "\x1b[36;3m"

    white           = "\x1b[37m"
    boldWhite       = "\x1b[37;1m"
    italicsWhite    = "\x1b[37;3m"

    reset           = "\x1b[0m"
   


    format = "Process:%(process)d||%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]: %(message)s"
    init =  "Process:%(process)d||%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]:"
    message = "%(message)s"

    FORMATS = {
        logging.DEBUG:  boldCyan + init + reset+ italicsBlue + message + reset,
        logging.INFO:   boldGreen + init + reset+ boldLightBlue + message + reset,
        logging.WARNING: boldYellow + init + reset+ italicsYellow + message + reset,
        logging.ERROR: red + format + reset, 
        logging.CRITICAL: boldRed + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

#===================================
# CONFIG FILE
#===================================
# LOGGING_CONFIG = { 
#     'version': 1,
#     'disable_existing_loggers': False,
#     #------------------------------
#     #  FORMAT
#     #------------------------------
#     'formatters': { 
#         'ai_colored_console':{
#             '()': CustomFormatter,
#             # 'format':'%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]: %(message)s',
#             'datefmt': '%Y-%m-%d %H:%M:%S'
#             },
#         'format_for_file':{
#             'format':'%(asctime)s-%(levelname)s [%(filename)s:%(funcName)s():%(lineno)d]: %(message)s',
#             'datefmt':'%Y-%m-%d %H:%M:%S'

#         }
#     },
#     #------------------------------
#     #  HANDLER
#     #------------------------------
#     'handlers': { 
#         'ai_console': { 
#             'level': 'DEBUG',
#             'formatter': 'ai_colored_console',
#             'class': 'logging.StreamHandler',
#             'stream': 'ext://sys.stdout',  # Default is stderr
#         },
#         # 'file': {
#         #     'level': 'DEBUG',
#         #     'class': 'logging.handlers.RotatingFileHandler',
#         #     'formatter': 'format_for_file',
#         #     'filename': 'ai_log/ai_log_file.log',
#         #     'mode':'a',
#         #     'maxBytes': 10485760,
#         #     'backupCount': 5
#         # }
#     },
#     #------------------------------
#     # LOGGERs
#     #------------------------------
#     'loggers': {
#         '': {
#            'level': 'DEBUG',
#            'handlers': ['ai_console'],
#            'propagate': False
#         },
#     }, 
#     #------------------------------
#     # ROOTS
#     #------------------------------
#     'root' : {
#         "level": "DEBUG",
#         "handlers": ["ai_console"]
#     }
# }


#===================================
# INIT LOGGER
#===================================

def setUpAiLogger(name=None):
    # if not os.path.isdir("ai_log"):
    #     os.makedirs("ai_log", exist_ok=True)

    # ---------------------
    # Create new logger
    # ---------------------
    logger = logging.getLogger("aiLog")
    # logging.config.dictConfig(LOGGING_CONFIG)
    # logger.setLevel(logging.DEBUG)

    # ---------------------
    # Set up handler
    # ---------------------
    # For file
    # file_handler = logging.handlers.RotatingFileHandler("ai_log/ai_log_file", maxBytes=10485760,backupCount=300, encoding='utf-8')
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(CustomFormatter())

    # For console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())

    # ---------------------
    # Add new console_handler
    # and turn off propagate
    # https://stackoverflow.com/questions/19561058/duplicate-output-in-simple-python-logging-configuration/19561320#19561320
    # ---------------------
    logger.propagate = False
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    # ---------------------
    # For root
    # ---------------------
    # logging.root.setLevel(logging.DEBUG)
    # logging.root.addHandler(file_handler)
    # logging.root.addHandler(console_handler)

    return logger



aiLogger = setUpAiLogger()

