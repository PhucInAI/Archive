#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=
#                       PLEASE DECLARE YOUR LIBRARY BELOW
#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=
import os 

import yaml
import torch

import ai_core.face_recognition.ArcFace.face_service as face_service
from ai_core.face_recognition.ArcFace.face_extractor.utils import utils


from ai_core.object_detection.yolov5_custom.od.data.datasets import letterbox
from ai_core.object_detection import ObjectDetection
from dynaconf import settings
from ai_core.object_detection.models.detection_model_name import ObjectDetectionModelName
from ai_core.object_detection.models.detection_config import ObjectDetectionConfig
from ai_core.face_recognition.ArcFace.face_extractor.utils.voting import decision
from ai_core.face_detection import FaceDetection
from dynaconf import settings
from ai_core.face_detection.models.facedetection_model_name import FaceDetectionModelName
from ai_core.face_detection.models.facedetection_config import FaceDetectionConfig
import torch
import time
from collections import OrderedDict

from ai_core.reid.info import human_detection_info, human_reid_info
from ai_core.face_recognition import FaceRecognition
from ai_core.reid.tracking_engine                           import TrackingEngine
from ai_core.face_recognition.models.facerecognition_model_name import FaceRecognitionModelName
from ai_core.utility.utils                                          import get_GPU_available
import torchvision.models as models
import torchvision.models._utils as _utils
from ai_core.utility.utils_image import extract_boxxy
from typing import List

import numpy as np 
import time
import cv2
#=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=--=

class HumanSearchEngine:
    def __init__(self, gpu_index, data_folder_path, model_path, is_tensorrt=False):
        
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #           DECLARE SOME VARIABLES FOR AI 
        #``
        #=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        cfg_hydranet= model_path + 'configs/Hydranet_engine.yml'
        with open(cfg_hydranet) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        torch.backends.cudnn.benchmark = cfg['cudnn_params']['benchmark']
        torch.backends.cudnn.enabled   = cfg['cudnn_params']['enabled']
        torch.backends.cudnn.deterministic = cfg['cudnn_params']['deterministic']
        #----------------------------------
        # Init general variable
        #----------------------------------

        if (not model_path.endswith('/')):
            model_path = model_path + '/'
        self.__model_path                   = model_path

        if (gpu_index < 0):
            gpu_index = 0

        gpu_idx,self.is_server              = get_GPU_available()
        
        self.gpu_is_available               = True if gpu_idx >=0 else False

        # print('wwwwwwwhuman',gpu_idx,self.is_server)
        device                              = "cuda:" + str(gpu_idx) if self.gpu_is_available  else "cpu"

        self.__device                       = torch.device(device)

        if (not data_folder_path.endswith('/')):
            data_folder_path = data_folder_path + '/'
        self.__folder_path                  = data_folder_path


        self.__face_folder_path = self.__folder_path + 'face/'
        os.makedirs(self.__face_folder_path, exist_ok=True)
        
        nn_config_path = f"{model_path}face_recognition/configs/face_engine.yml"
        self.cfg = {}
        with open(nn_config_path) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)

        weight_path = self.cfg['extractor_params']['weights']
        self.weight_version = weight_path.split('/')[-1]

        device, n_gpu_ids = utils.prepare_device(self.cfg['gpu_id'])
        # print('cfggggg',self.cfg)
        #face_det = face_service.init_face_detector(cfg, device)

        self.device = torch.device(self.__device)

        self.face_en = face_service.init_face_engine(self.cfg, self.device, n_gpu_ids, is_tensorrt=is_tensorrt, is_server=False, mode='inference', model_path=model_path)
        # self.face_en.init_face_registration()

        separate= False
        model_name  = ObjectDetectionModelName.Yolo_v5_custom
        cfg_hydranet= 'configs/Hydranet_engine.yml'
        if not separate:
            print("***********BackboneThread not separate*****************")
            self.backbone = ObjectDetection(cfg = cfg_hydranet, 
                                            weights_path= None,
                                            model_name  = model_name,
                                            device      = self.device,
                                            neckhead    = False,
                                            isTensorrt  = is_tensorrt,
                                            is_server   = self.is_server,
                                            model_path  = model_path)
        else:
            print("***********BackboneThread separate*****************")
                
            # replace_stride_with_dilation=self.replace_stride_with_dilation
            backbone = models.resnet50(pretrained=True,model_path=model_path)
            #backbone.eval()
            self.backbone = _utils.IntermediateLayerGetter(backbone, {'layer2': 1, 'layer3': 2, 'layer4': 3}).to(self.device).half()  # create


        model_name      = ObjectDetectionModelName.Yolo_v5_custom
        cfg_hydranet= 'configs/Hydranet_engine.yml'
        self.detector   = ObjectDetection(  cfg=cfg_hydranet,
                                            weights_path=None,
                                            device=self.device, 
                                            model_name=model_name,
                                            neckhead=True,
                                            isTensorrt  = is_tensorrt,
                                            is_server   = self.is_server,
                                            model_path=model_path)

        cfg_hydranet= 'configs/Hydranet_engine.yml'
        model_name_face      = FaceDetectionModelName.RetinaFace
        self.face_detector   = FaceDetection(cfg=cfg_hydranet,
                                        weights_path=None,
                                        model_name=model_name_face,
                                        device=self.device, isFloat = False, model_path=model_path)


        nn_config_path = model_path + '/reid/' + f"configs/person_search_engine.yml"
        with open(nn_config_path) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.person_reid = TrackingEngine(cfg, self.device, False, self.is_server, None, self.__model_path,  self.__folder_path)
                                        # (cfg, self.__device, False, self.is_server, None, self.__model_path, self.__data_path)

    def extract_face_align_feature(self,folder, file_name):
        # print(folder + '/face_al.png')
        print(folder + '/' + file_name + '_al.jpg')
        if (os.path.isfile(folder + '/' + file_name + '_al.jpg')):
            print('aaaaaaaaaaa')
        face_al = cv2.imread(folder + '/' + file_name + '_al.jpg')

        # print(face_al.shape)
        face = self.face_en.face_preprocessing(face_al, self.device)

        mid_outputs, feature = self.face_en.extract_feature(self.face_en.model, face) 
        # print('feature 0 ',feature)
        # print('feature', feature.shape)
        # print(feature.size(), mid_outputs['layer4'].size(), mid_outputs['layer3'].size())
        feature = self.face_en.get_feature_by_config(feature, use_fliplr=self.cfg['preprocessing_params']['use_fliplr'])

        feature, embedding, adpt_pooling = self.face_en.emd_preprocessing(mid_outputs['layer4'], feature)


        np_feature = np.load(folder + '/' + file_name + '.npy')
        np_feature = torch.from_numpy(np_feature).to(self.device)

        print(folder + '/' + file_name + '_emb.npy')
        np_emb = np.load(folder + '/' + file_name + '_emb.npy')
        np_emb = torch.from_numpy(np_emb).to(self.device)

        np_adptp = np.load(folder + '/' + file_name + '_adptp.npy')
        np_adptp = torch.from_numpy(np_adptp).to(self.device)

        # np.save('duc_phuong.npy', feature)

        # np.save('duc_phuong_emb.npy', embedding)

        # np.save('duc_phuong_adptp.npy', adpt_pooling)

        print('np_feature.shape', np_feature.shape, feature.shape)
        cosine = torch.nn.CosineSimilarity(dim=1)
        similarity = cosine(np_feature, feature)

        # print('duc feature',feature)
        # print('tan feature',np_feature)
        print('similarity feature ---- ', similarity)
        # print('feature',torch.equal(feature, np_feature ))


        # print('emb',torch.equal(embedding, np_emb))
        # print('duc emb',embedding)
        # print('tan emb',np_emb)
        print('np_emb.shape', np_emb.shape)
        similarity = cosine(np_emb, embedding)
        print('similarity embedding ---- ', similarity)

        # print('np_adptp',torch.equal(adpt_pooling, torch.from_numpy(np_adptp).to(self.device)))
        # print(embedding, np_emb)
        # print('duc',adpt_pooling)
        # print('tan',np_adptp)
        print('np_adptp.shape', np_adptp.shape)
        similarity = cosine(np_adptp, adpt_pooling)
        print('similarity adpt_pooling ---- ', similarity)

    def extract_person_info(self, image, isFile = True):
        meta_ret = {}
        meta_ret['is_person'] = False
        meta_ret['meta'] = {}
        meta_ret['meta']['person_id'] = 'Unknown'
        meta_ret['meta']['feature_path'] = './'
        meta_ret['meta']['visualize_path'] = './'
        meta_ret['meta']['box_xywh'] = ''
        if (os.path.isfile(image) or not isFile):
            raw_frame=[]
            
            if (isFile):
                frame = cv2.imread(image)
            else:
                frame = image
            raw_frame.append(frame)
            with torch.no_grad():
                features = list(self.backbone.backbone(raw_frame))

                for i in range(len(features)):
                    features[i]=features[i].half()
                    
                objDetectionResults = self.detector.detect(raw_frame, features)

                humanInfo = self.convertObjectDetection(raw_frame, objDetectionResults)


                feature_path   = None
                visualize_path = None
                recognize      = [None,'0.0']
                box = None
                if len(humanInfo) > 0:
                    feature, person_raw, feature_path, visualize_path, recognize, box \
                    = self.person_reid.person_registration(self.__folder_path + '/reid/', frame, humanInfo)                    
                    # print('aaaaaaaaaaa', feature_path, visualize_path, recognize, box)
                    if (box is not None):
                        meta_ret['is_person'] = True

                        meta_ret['meta']['feature_path']   = feature_path
                        meta_ret['meta']['visualize_path'] = visualize_path
                        box[2] = box[2] - box[0]
                        box[3] = box[3] - box[1]
                        meta_ret['meta']['box_xywh'] = box
                        meta_ret['meta']['person_id'] = recognize[0]
                        # print('aaaaaaaaaaa', meta_ret)

        return meta_ret

    def convertObjectDetection(self, frames, results):
        humanInfo = []
        for i in range(len(results.index_list)):
            for r in results.lcxyxy_list[i]:
                if(r != None):
                    ih,iw, _ = frames[i].shape
                    label, conf, c1, c2, text = extract_boxxy(r, iw, ih)
                # print(label, c1, c2)
                    if (label == 'person'):
                        # print('texxxtttt', label,r, c1,c2)
                        (xl, yl) , (xr, yr) = c1, c2

                        xl = 0 if xl < 0 else xl
                        yl = 0 if yl < 0 else yl
                        xr = iw - 1 if xr > iw else xr
                        yr = ih - 1 if yr > ih else yr

                        humanInfo.append(human_detection_info.HumanDetectionInfo(0,[xl,yl,xr,yr],[conf,0,label]))

            break
        return humanInfo

    def extract_face_info(self,image, isFile = True):

        meta_ret = {}
        meta_ret['is_face'] = False
        meta_ret['meta'] = {}
        meta_ret['meta']['version'] = self.weight_version 
        meta_ret['meta']['face_id'] = 'Unknown'
        meta_ret['meta']['feature_path'] = './'
        meta_ret['meta']['face_align_image_path'] = './'
        meta_ret['meta']['box_xywh'] = ''
        meta_ret['meta']['face_similarity'] = 0.0
        if (os.path.isfile(image) or not isFile):
            raw_frame=[]
            if (isFile):
                frame = cv2.imread(image)
            else:
                frame = image

            # print(frame.dtype, type(frame))
            # frame = cv2.resize(frame,(640,480))
            # cv2.imshow('Frame', frame)
            raw_frame.append(frame)
            with torch.no_grad():
                features = list(self.backbone.backbone(raw_frame))

                for i in range(len(features)):
                    features[i]=features[i].half()
                # print(features, features[0].dtype)
                for lay in features.copy():
                    if lay.shape[1] == 256:
                        features.remove(lay)

                self.face_detector._face_detection.setZones([[(0,0),(0,1),(1,1),(1,0)]],raw_frame[0])
                self.face_detector._face_detection.confidence = 0.5
                self.face_detector._face_detection.w_thres = 60
                self.face_detector._face_detection.h_thres = 60
                results,dets   = self.face_detector.detect(raw_frame, features)
                feature_path   = None
                visualize_path = None
                recognize      = ['Unknown','0.0']
                box = None
                landmarks = None
                if (len(results.face_detections_list)):
                    feature_path, register_emb_path, register_adpt_pooling_path, visualize_path, recognize, box, landmarks \
                    = self.face_en.face_registration2(self.__face_folder_path, frame, results.face_detections_list[0]) 
                                
                    # print('aaaaaaaaaaa', feature_path, visualize_path, recognize, recognize[0], box)
                    if (box is not None):
                        meta_ret['is_face'] = True
                        meta_ret['meta']['feature_path'] = []
                        meta_ret['meta']['feature_path'].append(feature_path)
                        meta_ret['meta']['feature_path'].append(register_emb_path)
                        meta_ret['meta']['feature_path'].append(register_adpt_pooling_path)
                        meta_ret['meta']['face_align_image_path'] = visualize_path
                        box[2] = box[2] - box[0]
                        box[3] = box[3] - box[1]
                        meta_ret['meta']['landmarks'] = landmarks
                        meta_ret['meta']['box_xywh'] = box
                        # print('recognize----------- ',recognize)
                        meta_ret['meta']['face_id'] = recognize[0]
                        meta_ret['meta']['face_similarity'] = recognize[0]
                        # print('aaaaaaaaaaa', meta_ret)

                        log_file = self.__face_folder_path + str(time.time()) + '_meta.txt'
                        with open(log_file, 'w') as f:
                            print(str(meta_ret), file=f)

                        meta_ret['meta']['face_metadata_path'] = log_file
        # print('aaaaaaaaaaa', meta_ret)
        return meta_ret

    def __to_device(self, registers, embedding_registers, adpt_pooling_registers, use_fliplr=False):


        registers = torch.from_numpy(np.array(registers))
        
        # print('aa',registers.shape)

        embedding_registers = torch.from_numpy(np.array(embedding_registers))

        adpt_pooling_registers = torch.from_numpy(np.array(adpt_pooling_registers))

        # if (use_fliplr):
        w,h,c = 0,0,0
        if (registers.dim() == 3):
            w,h,c = registers.shape
        elif (registers.dim() == 2):
            w,h = registers.shape
        # print(w,h,c)
        if (h == 512 or c == 512) and use_fliplr:
            registers = [self.face_en.get_feature_by_config(feature, use_fliplr=use_fliplr) for feature in registers]

            registers = torch.cat(registers)

        # print('vv',registers.shape)

        registers = registers.to(self.device)

        register_embs = torch.unsqueeze(registers, 0).to(self.device).float()

        embedding_registers = embedding_registers.to(self.device)

        adpt_pooling_registers = adpt_pooling_registers.to(self.device)

        return registers, register_embs, embedding_registers, adpt_pooling_registers

    def search_unknown_faces(self, 
                                feature_input_path, 
                                detected_unknown_features,
                                similarity_percent, is_use_emd=False):
        feature = []
        embedding = []
        adpt_pooling = []
        feature_emb = []

        id_ret_list = []

        recognize = []
        conf = []
        try:
        
            if (len(feature_input_path) == 3 and os.path.isfile(feature_input_path[0]) and os.path.isfile(feature_input_path[1]) and os.path.isfile(feature_input_path[2]) ):
                feature.append(np.load(feature_input_path[0]))
                embedding.append(np.load(feature_input_path[1]))
                adpt_pooling.append(np.load(feature_input_path[2]))
                feature, feature_emb, embedding, adpt_pooling = self.__to_device(feature, embedding, adpt_pooling, self.cfg['preprocessing_params']['use_fliplr'])
            ids            = []
            u_features     = []
            u_feature_embs = []
            u_embeddings   = []
            u_adpt_poolings= []
            for u in detected_unknown_features:
                if (len(u['feature_path']) == 3 and os.path.isfile(u['feature_path'][0]) and os.path.isfile(u['feature_path'][1]) and os.path.isfile(u['feature_path'][2])):
                    u_features.append(np.load(u['feature_path'][0]))
                    u_embeddings.append(np.load(u['feature_path'][1]))
                    u_adpt_poolings.append(np.load(u['feature_path'][2]))
                    ids.append(u['feature_id'])

            u_features, u_feature_embs, u_embeddings, u_adpt_poolings = self.__to_device( u_features, u_embeddings, u_adpt_poolings, self.cfg['preprocessing_params']['use_fliplr'])
            # print('--feature.shape', u_features.shape)
            # print('--embedding.shape', u_embeddings.shape)
            # print('--adpt_pooling.shape', u_adpt_poolings.shape)
            # u_features_embs = torch.unsqueeze(torch.from_numpy(np.array(u_features)), 0).to(self.device).float()
            # print(feature,feature.shape)

            if (len(ids)):

                scores = self.face_en.calculate_score(feature, u_feature_embs, None, self.device)

                recognize, conf = decision.rule2(scores, ids, 0, 
                                                embedding, adpt_pooling,
                                                u_embeddings, u_adpt_poolings,
                                                alpha=self.cfg['postprocessing_params']['alpha'],
                                                use_emd=is_use_emd,
                                                similarity_threshold=similarity_percent/100.0,
                                                emd_threshold=similarity_percent/100.0, 
                                                topK=len(ids),
                                                method=self.cfg['postprocessing_params']['method'])
        except Exception as e:
            print("------- Face search engine search_unknown_faces Exception " + str(e))

        print(recognize, conf)
        return recognize, conf

    def update_face_bank(self,face_bank_list=[],config="add"):
                           #[{ 
                           #     "id" "duc",
                           #     "images": ["ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3982.jpg",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3983.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3984.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3985.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Duc/IMG_3986.jpg"   
                           #                 ],
                           #     "features": ["ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3982.npy",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3983.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3984.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3985.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Duc/IMG_3986.npy"   
                           #                 ]
                           #  },
                           #  {"id" "Bang",
                           #     "images": ["ai_core/face_recognition/ArcFace/resources/registers/images/Bang/1.jpg",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/2.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/3.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/4.jpg",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/images/Bang/5.jpg"   
                           #                 ],
                           #     "features": ["ai_core/face_recognition/ArcFace/resources/registers/features/Bang/1.npy",
                           #                 "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/2.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/3.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/4.npy",
                           #                  "ai_core/face_recognition/ArcFace/resources/registers/features/Bang/5.npy" 
                           #   }]*/
        if config=="add":
            self.face_en.add_face(face_bank_list, self.cfg['register_params']['use_mean'], self.cfg['preprocessing_params']['use_fliplr'])
        elif (config=="delete"):
            self.face_en.delete_face(face_bank_list, self.cfg['register_params']['use_mean'], self.cfg['preprocessing_params']['use_fliplr'])

        

    def load_registers(self,):
        self.face_en.load_registers('./',self.cfg['register_params']['use_mean'], self.cfg['preprocessing_params']['use_fliplr'])

    

