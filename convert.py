#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

# COLMAP 변환 스크립트 - MipNerF 360 레포지토리의 쉘 스크립트 기반
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')  # GPU 사용 안함
parser.add_argument("--skip_matching", action='store_true')  # Feature matching 건너뛰기
parser.add_argument("--source_path", "-s", required=True, type=str)  # 데이터 경로
parser.add_argument("--camera", default="OPENCV", type=str)  # 카메라 모델 타입
parser.add_argument("--colmap_executable", default="", type=str)  # COLMAP 실행 파일 경로
parser.add_argument("--resize", action="store_true")  # 이미지 리사이징 수행
parser.add_argument("--magick_executable", default="", type=str)  # ImageMagick 실행 파일 경로
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction - 이미지에서 특징점(SIFT) 추출
    # 각 이미지에서 구별 가능한 특징점들을 찾아냄
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching - 이미지들 간의 특징점 매칭
    # 여러 이미지에서 같은 3D 포인트를 찾아냄
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db"
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment - 카메라 포즈와 3D 포인트 최적화
    # 매칭된 특징점들을 이용해 카메라 위치/회전과 3D 구조를 계산
    # tolerance를 낮춰서 번들 조정 속도를 향상
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion - 이미지 왜곡 보정
## 실제 카메라 렌즈의 왜곡을 제거하고 이상적인 핀홀 카메라 모델로 변환
## 이를 통해 3DGS 학습이 더 정확해짐
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# sparse 폴더 구조 정리 - sparse/0/ 디렉토리로 파일들 이동
files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if(args.resize):
    print("Copying and resizing...")

    # 다양한 해상도의 이미지 생성 (멀티스케일 학습용)
    # images_2: 50% 크기, images_4: 25% 크기, images_8: 12.5% 크기
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    
    files = os.listdir(args.source_path + "/images")
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        # 50% 크기로 리사이즈
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 25% 크기로 리사이즈
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 12.5% 크기로 리사이즈
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")
