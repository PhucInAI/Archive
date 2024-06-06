"""Preprocess raw data to YOLO"""


# pylint: disable=invalid-name, too-many-locals, import-error


import os
import glob
import shutil
import json
import pandas as pd
import numpy as np

np.random.seed(1)
ROOT_PATH = 'data/raw'
DST_PATH = 'data/yolo'


def preprocess_dubska():
    """Transform from raw to YOLO format for Dubska"""
    dataset_part_lst = ['dataset1', 'dataset2']
    for dataset_part in dataset_part_lst:
        # ----------------------------------------------------------------
        # Prepare data and paths
        # ----------------------------------------------------------------
        data_folder = os.path.join(ROOT_PATH, 'Dubska_QR', dataset_part)
        bitmap_folder = os.path.join(ROOT_PATH, 'Dubska_QR', 'bitmaps')
        quote = pd.read_csv(os.path.join(ROOT_PATH, 'Dubska_QR','quote.csv'), sep=';')

        img_path_lst = glob.glob(os.path.join(data_folder, '*.jpg'))

        dst_img_path = os.path.join(DST_PATH, 'Dubska_QR', dataset_part, 'images')
        dst_bitmap_path = os.path.join(DST_PATH, 'Dubska_QR', dataset_part, 'bitmaps')
        dst_content_path = os.path.join(DST_PATH, 'Dubska_QR', dataset_part, 'contents')

        # ----------------------------------------------------------------
        # Copy image to folder
        # ----------------------------------------------------------------
        os.makedirs(dst_img_path, exist_ok=True)

        for img_path in img_path_lst:
            shutil.copy(img_path, dst_img_path)

        # ----------------------------------------------------------------
        # Copy bitmaps with corresponding images
        # ----------------------------------------------------------------
        os.makedirs(dst_bitmap_path, exist_ok=True)
        anno = pd.read_csv(os.path.join(data_folder, 'anotation.csv'), sep=';', header=None)
        for _, row in anno.iterrows():
            img_name = row[0]
            bitmap_id = row[1]
            bitmap_size = row[2]
            bitmap_name = f'cite_{str(bitmap_id).zfill(2)}_{bitmap_size}_small.png'
            shutil.copy(os.path.join(bitmap_folder, bitmap_name),
                        os.path.join(dst_bitmap_path, img_name))

        # ----------------------------------------------------------------
        # Make quotes
        # ----------------------------------------------------------------
        os.makedirs(dst_content_path, exist_ok=True)
        for _, anno_row in anno.iterrows():
            img_name = anno_row[0]
            bitmap_id = anno_row[1]

            quote_by_id = quote[quote['ID']==bitmap_id].iloc[0]

            content = {
                        'quote': quote_by_id['quote'],
                        'author': quote_by_id['author'],
                      }
            content_name = img_name.split('.png')[0] + '.json'
            content_path = os.path.join(dst_content_path, content_name)
            with open(content_path, 'w', encoding='utf-8') as json_file:
                json.dump(content, json_file, indent=4)


# def preprocess_szentandrasi():
#     """Transform from raw to YOLO format for Szentandrasi"""
#     # ----------------------------------------------------------------
#     # Prepare data and paths
#     # ----------------------------------------------------------------
#     data_folder = os.path.join(ROOT_PATH, 'Szentandrasi_QR', 'datasets')
#     bitmap_folder = os.path.join(ROOT_PATH, 'Szentandrasi_QR', 'bitmaps')

#     img_path_lst = glob.glob(os.path.join(data_folder, '*', '*.jpg')) + \
#                    glob.glob(os.path.join(data_folder, '*', '*.JPG'))

#     dst_img_path = os.path.join(DST_PATH, 'Szentandrasi_QR', 'images')
#     dst_label_path = os.path.join(DST_PATH, 'Szentandrasi_QR', 'labels')
#     dst_bitmap_path = os.path.join(DST_PATH, 'Szentandrasi_QR', 'bitmaps')
#     dst_content_path = os.path.join(DST_PATH, 'Szentandrasi_QR', 'contents')

#     # ----------------------------------------------------------------
#     # Copy image to folder
#     # ----------------------------------------------------------------
#     os.makedirs(dst_img_path, exist_ok=True)

#     for img_path in img_path_lst:
#         img_name = os.path.basename(img_path)[:-4] + '.jpg' # normalize extensions
#         shutil.copy(img_path, os.path.join(dst_img_path, img_name))


    # # ----------------------------------------------------------------
    # # Copy labels
    # # ----------------------------------------------------------------
    # os.makedirs(dst_label_path, exist_ok=True)

    # for img_path in img_path_lst:
    #     img = cv2.imread(img)
    #     img_h, img_w = img.shape[:2]

    #     label_path = img_path[-4] + '_annotated.csv'
    #     label_df = pd.read_csv(label_path, sep=',')
    #     for _, row in label_df.iterrows():
    #         x_c = row[label_df.columns[0]]
    #         y_c = row['Y']
    #         r = row['Radius']
    #     img_name = os.path.basename(img_path)[:-4] + '.jpg' # normalize extensions
    #     shutil.copy(img_path, os.path.join(dst_img_path, img_name))

    # # ----------------------------------------------------------------
    # # Copy bitmaps with corresponding images
    # # ----------------------------------------------------------------
    # os.makedirs(dst_bitmap_path, exist_ok=True)
    # anno = pd.read_csv(os.path.join(data_folder, 'anotation.csv'), sep=';', header=None)
    # for _, row in anno.iterrows():
    #     img_name = row[0]
    #     bitmap_id = row[1]
    #     bitmap_size = row[2]
    #     bitmap_name = f'cite_{str(bitmap_id).zfill(2)}_{bitmap_size}_small.png'
    #     shutil.copy(os.path.join(bitmap_folder, bitmap_name),
    #                 os.path.join(dst_bitmap_path, img_name))

    # # ----------------------------------------------------------------
    # # Make quotes
    # # ----------------------------------------------------------------
    # os.makedirs(dst_content_path, exist_ok=True)
    # for _, anno_row in anno.iterrows():
    #     img_name = anno_row[0]
    #     bitmap_id = anno_row[1]

    #     quote_by_id = quote[quote['ID']==bitmap_id].iloc[0]

    #     content = {
    #                 'quote': quote_by_id['quote'],
    #                 'author': quote_by_id['author'],
    #                 }
    #     content_name = img_name.split('.png')[0] + '.json'
    #     content_path = os.path.join(dst_content_path, content_name)
    #     with open(content_path, 'w', encoding='utf-8') as json_file:
    #         json.dump(content, json_file, indent=4)


preprocess_dubska()
