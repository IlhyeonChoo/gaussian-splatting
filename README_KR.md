# Gaussian Splatting 설치 및 사용 가이드

## 설치 완료! ✓

모든 필요한 패키지가 설치되었습니다:
- PyTorch 2.5.1 (CUDA 12.1)
- simple-knn
- diff-gaussian-rasterization
- fused-ssim
- opencv-python
- plyfile, tqdm, joblib

## 사용 방법

### 1. 가상 환경 활성화

매번 사용하기 전에 가상 환경을 활성화해야 합니다:

```bash
cd /home/ilhyeonchu/ReCompose3D/3DGS/gaussian-splatting
source venv/bin/activate
```

### 2. 데이터 준비

두 가지 방법으로 데이터를 준비할 수 있습니다:

#### 방법 A: 자신의 이미지/영상 사용 (권장)

1. 이미지들을 폴더에 준비:
```bash
mkdir -p data/my_scene/input
# 자신의 이미지들을 data/my_scene/input/ 폴더에 복사
```

영상에서 이미지를 추출해서 사용할 경우:
```bash
source venv/bin/activate
python extract_video_frames.py \
  --video_path /path/to/input.mp4 \
  --output_dir data/my_scene/frames \
  --mode both \
  --target_fps 2 \
  --scale 0.5 \
  --custom_format jpg \
  --jpeg_quality 90
```

위 명령은 다음을 동시에 생성합니다:
- `data/my_scene/frames/original/`: 원본 해상도 프레임
- `data/my_scene/frames/custom/`: 사용자 지정 화질(해상도/포맷/품질) 프레임

자주 쓰는 옵션:
- `--mode original`: 원본 화질만 추출
- `--mode custom`: 사용자 지정 화질만 추출
- `--width 1280` 또는 `--height 720`: 원하는 해상도로 고정
- `--every_nth 10`: 10프레임마다 1장 저장
- `--target_fps 2`: 초당 2장 저장 (`--every_nth`보다 우선)

2. COLMAP으로 카메라 포즈 추출 (자동):
```bash
python convert.py -s data/my_scene
```

이 과정은 다음을 수행합니다:
- 이미지에서 특징점 추출
- 카메라 위치와 각도 계산
- 3D 포인트 클라우드 생성

기본 동작은 `--colmap_device auto` 입니다.
- `auto`: GPU를 먼저 시도하고 실패하면 CPU로 다시 시도
- `gpu`: GPU만 사용하고 실패 시 종료
- `cpu`: 처음부터 CPU만 사용

설치된 COLMAP이 `without CUDA` 빌드라면 `auto`는 GPU 시도를 건너뛰고 CPU로 바로 진행합니다.

COLMAP 4.1 기준으로 matcher, 특징점, 매칭, mapper, undistortion 옵션도 조절할 수 있습니다.
- 동영상 프레임처럼 순서가 있는 이미지: `--colmap_matcher sequential --sequential_overlap 10`
- 메모리가 부족한 경우: `--feature_max_image_size 1600 --sift_max_num_features 4096 --matching_max_num_matches 10000`
- 저텍스처/등록 실패가 많은 경우: `--sift_max_num_features 16384 --sift_peak_threshold 0.003 --guided_matching 1`
- 서로 다른 카메라/줌/크롭 이미지가 섞인 경우: `--single_camera 0`
- 학습 입력 해상도를 미리 제한하고 싶은 경우: `--undistort_max_image_size 1600`

예시:
```bash
python convert.py -s data/my_scene --colmap_device auto
python convert.py -s data/my_scene --colmap_device cpu
python convert.py -s data/my_scene --colmap_matcher sequential --sequential_overlap 10
```

#### 방법 B: 샘플 데이터 다운로드

```bash
# MipNeRF360 샘플 데이터 다운로드 (예: garden scene)
mkdir -p data
cd data
# 샘플 데이터 링크는 README.md의 상단에 있습니다
```

### 3. 학습 (Training)

기본 학습:
```bash
python train.py -s data/my_scene
```

고급 옵션:
```bash
# 더 빠른 학습 (sparse adam 사용)
python train.py -s data/my_scene --optimizer_type sparse_adam

# 흰색 배경 사용 (투명 물체용)
python train.py -s data/my_scene -w

# 특정 해상도로 학습 (1=원본, 2=1/2, 4=1/4)
python train.py -s data/my_scene -r 2

# 학습 횟수 조정 (기본: 30,000)
python train.py -s data/my_scene --iterations 10000
```

학습 중에는:
- `output/` 폴더에 결과가 저장됩니다
- 학습 진행 상황이 콘솔에 출력됩니다
- 체크포인트가 자동으로 저장됩니다

### 4. 렌더링

학습된 모델로 이미지 렌더링:
```bash
python render.py -m output/<모델_폴더_이름>
```

결과는 `output/<모델_폴더_이름>/train/renders/` 와 `test/renders/` 에 저장됩니다.

### 5. 평가

렌더링 품질 측정 (PSNR, SSIM, LPIPS):
```bash
python metrics.py -m output/<모델_폴더_이름>
```

## 빠른 시작 예제

```bash
# 1. 가상환경 활성화
cd /home/ilhyeonchu/ReCompose3D/3DGS/gaussian-splatting
source venv/bin/activate

# 2. 자신의 이미지로 데이터 준비
mkdir -p data/test_scene/input
# 이미지들을 data/test_scene/input/에 복사한 후:
python convert.py -s data/test_scene --colmap_device auto

# 3. 학습 (약 30분 소요, GPU에 따라 다름)
python train.py -s data/test_scene --iterations 7000

# 4. 렌더링
python render.py -m output/<생성된_폴더>

# 5. 결과 확인
# output/<생성된_폴더>/train/renders/ 에서 렌더링된 이미지를 확인
```

## 팁과 요령

### 데이터 준비
- 최소 30장 이상의 이미지 권장
- 다양한 각도에서 촬영
- 적절한 조명 (너무 밝거나 어둡지 않게)
- 카메라를 천천히 움직이며 촬영
- 흔들림 최소화

### 학습 최적화
- VRAM이 부족하면: `--resolution 2` 또는 `--resolution 4` 사용
- 빠른 테스트: `--iterations 7000`
- 고품질: `--iterations 30000` (기본값)

### 문제 해결
- CUDA out of memory: 해상도를 낮추거나 `--data_device cpu` 사용
- 학습이 너무 느림: `--optimizer_type sparse_adam` 사용
- 결과가 흐릿함: 더 많은 iteration으로 학습

## 추가 기능

### 실시간 뷰어 (선택사항)
SIBR 뷰어를 사용하면 실시간으로 3D 씬을 탐색할 수 있습니다.
설치 방법은 원본 README.md의 "Interactive Viewers" 섹션 참조.

### 깊이 정규화
더 나은 품질을 위해 깊이 맵 사용:
```bash
# depth maps 생성 후
python train.py -s data/my_scene -d data/my_scene/depths
```

## 자세한 정보

더 많은 옵션과 고급 기능은 원본 README.md를 참조하세요.

## 문제 발생 시

```bash
# 모든 것을 다시 설치하고 싶다면:
cd /home/ilhyeonchu/ReCompose3D/3DGS/gaussian-splatting
rm -rf venv
python3 -m venv venv
source venv/bin/activate
# 그리고 위의 설치 과정을 다시 따라하세요
```
