#!/bin/bash
# Gaussian Splatting 빠른 시작 스크립트

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 상대 경로만 허용 (절대 경로, ~ 경로 금지)
is_relative_path() {
    local p="$1"
    [[ -n "$p" && "$p" != /* && "$p" != ~* ]]
}

# 이미지 파일 목록 수집
collect_image_files() {
    local dir="$1"
    find "$dir" -maxdepth 1 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort -V
}

# 균등 샘플링으로 최대 개수 제한하여 복사
copy_images_with_limit() {
    local src_dir="$1"
    local dst_dir="$2"
    local max_images="$3"

    mapfile -t files < <(collect_image_files "$src_dir")
    local total="${#files[@]}"

    if [ "$total" -eq 0 ]; then
        echo "오류: 이미지 파일(jpg/jpeg/png/webp)이 없습니다: $src_dir"
        return 1
    fi

    mkdir -p "$dst_dir"
    find "$dst_dir" -maxdepth 1 -type f -delete

    if [ -z "$max_images" ] || [ "$max_images" -le 0 ] || [ "$max_images" -ge "$total" ]; then
        for f in "${files[@]}"; do
            cp "$f" "$dst_dir/"
        done
        echo "[INFO] 이미지 복사: $total / $total (제한 없음)"
        return 0
    fi

    for ((i=0; i<max_images; i++)); do
        idx=$(( i * total / max_images ))
        cp "${files[$idx]}" "$dst_dir/"
    done
    echo "[INFO] 이미지 샘플링 복사: $max_images / $total"
}

# 중복 없는 출력 디렉토리 경로 생성
next_available_output_dir() {
    local base_dir="$1"
    local candidate="$base_dir"
    local idx=2
    while [ -e "$candidate" ]; do
        candidate="${base_dir}_v${idx}"
        idx=$((idx + 1))
    done
    echo "$candidate"
}

# 가상환경 활성화
echo "가상환경 활성화 중..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[WARN] venv/bin/activate를 찾을 수 없습니다. 시스템 Python을 사용합니다."
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "오류: python 또는 python3 명령을 찾을 수 없습니다."
    exit 1
fi
echo "[INFO] Python 실행기: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# 함수: 데이터 준비
prepare_data() {
    echo ""
    echo "=== 데이터 준비 ==="
    echo "이미지 폴더의 상대 경로를 입력하세요 (예: data/raw_images):"
    read -p "상대 경로: " img_folder

    if ! is_relative_path "$img_folder"; then
        echo "오류: 절대 경로는 사용할 수 없습니다. 상대 경로만 입력하세요."
        return 1
    fi
    
    if [ ! -d "$img_folder" ]; then
        echo "오류: 폴더를 찾을 수 없습니다: $img_folder"
        return 1
    fi
    
    echo "출력 경로(상대 경로)를 입력하세요 (예: data/my_scene):"
    read -p "상대 경로: " scene_path

    if ! is_relative_path "$scene_path"; then
        echo "오류: 절대 경로는 사용할 수 없습니다. 상대 경로만 입력하세요."
        return 1
    fi

    echo ""
    echo "최대 입력 이미지 수를 입력하세요 (기본: 0, 0=제한 없음)"
    echo "참고: 여기서 줄이면 COLMAP 자체 카메라가 줄어듭니다."
    read -p "최대 이미지 수: " max_images
    max_images=${max_images:-0}

    if ! [[ "$max_images" =~ ^[0-9]+$ ]]; then
        echo "오류: 최대 이미지 수는 0 이상의 정수여야 합니다."
        return 1
    fi

    echo ""
    echo "COLMAP 실행 모드를 선택하세요:"
    echo "1) auto (기본, GPU 시도 후 실패하면 CPU로 재시도)"
    echo "2) gpu  (GPU만 사용, 실패 시 종료)"
    echo "3) cpu  (처음부터 CPU만 사용)"
    read -p "선택 (1-3, 기본: 1): " colmap_mode_choice

    case "${colmap_mode_choice:-1}" in
        1) colmap_device="auto" ;;
        2) colmap_device="gpu" ;;
        3) colmap_device="cpu" ;;
        *)
            echo "잘못된 선택. 기본값(auto) 사용"
            colmap_device="auto"
            ;;
    esac

    echo ""
    echo "COLMAP 프리셋을 선택하세요:"
    echo "1) default    (기본 exhaustive matcher)"
    echo "2) video      (연속 프레임용 sequential matcher)"
    echo "3) low-memory (낮은 해상도/특징점/매칭 수)"
    echo "4) hard-scene (특징점 증가 + guided matching)"
    read -p "선택 (1-4, 기본: 1): " colmap_preset_choice

    colmap_args=()
    case "${colmap_preset_choice:-1}" in
        1) ;;
        2) colmap_args=(--colmap_matcher sequential --sequential_overlap 10) ;;
        3) colmap_args=(--feature_max_image_size 1600 --sift_max_num_features 4096 --matching_max_num_matches 10000) ;;
        4) colmap_args=(--sift_max_num_features 16384 --sift_peak_threshold 0.003 --guided_matching 1) ;;
        *)
            echo "잘못된 선택. 기본값(default) 사용"
            ;;
    esac

    mkdir -p "$scene_path/input"
    copy_images_with_limit "$img_folder" "$scene_path/input" "$max_images" || return 1
    
    echo ""
    echo "[INFO] COLMAP 실행 모드: $colmap_device"
    if [ "${#colmap_args[@]}" -gt 0 ]; then
        echo "[INFO] COLMAP 추가 옵션: ${colmap_args[*]}"
    else
        echo "[INFO] COLMAP 추가 옵션: 없음"
    fi
    echo "COLMAP으로 카메라 포즈 추출 중... (시간이 걸릴 수 있습니다)"
    convert_cmd=("$PYTHON_BIN" convert.py -s "$scene_path" --colmap_device "$colmap_device")
    convert_cmd+=("${colmap_args[@]}")
    "${convert_cmd[@]}" || return 1

    if [ -f "$scene_path/sparse/0/images.bin" ]; then
        cam_count=$("$PYTHON_BIN" - "$scene_path/sparse/0/images.bin" <<'PY'
import sys
from utils.read_write_model import read_images_binary
print(len(read_images_binary(sys.argv[1])))
PY
)
        echo "[INFO] COLMAP 등록 카메라 수: $cam_count"
    fi
    
    echo ""
    echo "✓ 데이터 준비 완료: $scene_path"
}

# 함수: 동영상에서 프레임 추출
extract_frames_from_video() {
    echo ""
    echo "=== 동영상 -> 이미지 추출 ==="

    if [ ! -f "extract_video_frames.py" ]; then
        echo "오류: extract_video_frames.py 파일이 없습니다."
        return 1
    fi

    echo "동영상 파일의 상대 경로를 입력하세요 (예: data/videos/input.mp4):"
    read -p "상대 경로: " video_path

    if ! is_relative_path "$video_path"; then
        echo "오류: 절대 경로는 사용할 수 없습니다. 상대 경로만 입력하세요."
        return 1
    fi

    if [ ! -f "$video_path" ]; then
        echo "오류: 파일을 찾을 수 없습니다: $video_path"
        return 1
    fi

    echo "프레임 출력 경로(상대 경로)를 입력하세요 (예: data/my_scene/frames):"
    read -p "상대 경로: " output_dir
    output_dir=${output_dir:-data/my_scene/frames}

    if ! is_relative_path "$output_dir"; then
        echo "오류: 절대 경로는 사용할 수 없습니다. 상대 경로만 입력하세요."
        return 1
    fi

    echo ""
    echo "추출 모드를 선택하세요:"
    echo "1) 원본 + 사용자 화질 모두 생성 (추천)"
    echo "2) 원본 화질만 생성"
    echo "3) 사용자 화질만 생성"
    read -p "선택 (1-3): " mode_choice

    case $mode_choice in
        1) mode="both" ;;
        2) mode="original" ;;
        3) mode="custom" ;;
        *) echo "잘못된 선택. 기본값(both) 사용"; mode="both" ;;
    esac

    read -p "저장 FPS 입력 (기본: 2): " target_fps
    target_fps=${target_fps:-2}

    if [ "$mode" = "custom" ] || [ "$mode" = "both" ]; then
        read -p "가로 해상도 입력 (비우면 scale 사용): " custom_width
        read -p "세로 해상도 입력 (비우면 비율 유지): " custom_height
        read -p "해상도 배율 입력 (기본: 1.0): " custom_scale
        custom_scale=${custom_scale:-1.0}
        read -p "이미지 품질(1-100, 기본: 90): " jpeg_quality
        jpeg_quality=${jpeg_quality:-90}
        read -p "커스텀 포맷(jpg/png/webp, 기본: jpg): " custom_format
        custom_format=${custom_format:-jpg}
    fi

    mkdir -p "$output_dir"

    echo ""
    echo "프레임 추출 시작..."

    cmd=("$PYTHON_BIN" extract_video_frames.py --video_path "$video_path" --output_dir "$output_dir" --mode "$mode" --target_fps "$target_fps")

    if [ "$mode" = "custom" ] || [ "$mode" = "both" ]; then
        if [ -n "$custom_width" ]; then
            cmd+=(--width "$custom_width")
        fi
        if [ -n "$custom_height" ]; then
            cmd+=(--height "$custom_height")
        fi
        cmd+=(--scale "$custom_scale" --custom_format "$custom_format" --jpeg_quality "$jpeg_quality")
    fi

    "${cmd[@]}"

    echo ""
    echo "✓ 프레임 추출 완료!"
    echo "결과 위치: $output_dir"
    echo " - 원본: $output_dir/original"
    echo " - 사용자 화질: $output_dir/custom"
}

# 함수: 학습
train_model() {
    echo ""
    echo "=== 모델 학습 ==="
    echo "데이터 폴더의 상대 경로를 입력하세요 (예: data/my_scene):"
    read -p "상대 경로: " scene_path

    if ! is_relative_path "$scene_path"; then
        echo "오류: 절대 경로는 사용할 수 없습니다. 상대 경로만 입력하세요."
        return 1
    fi

    if [ ! -d "$scene_path" ]; then
        echo "오류: 데이터 폴더를 찾을 수 없습니다: $scene_path"
        return 1
    fi

    echo ""
    read -p "학습 횟수(iterations)를 입력하세요 (기본: 30000): " iterations
    iterations=${iterations:-30000}

    if ! [[ "$iterations" =~ ^[0-9]+$ ]] || [ "$iterations" -lt 1 ]; then
        echo "오류: iterations는 1 이상의 정수여야 합니다."
        return 1
    fi

    echo "학습에 사용할 최대 카메라 수를 입력하세요 (기본: 120, 0=제한 없음)"
    read -p "최대 카메라 수: " max_train_cameras
    max_train_cameras=${max_train_cameras:-120}
    if ! [[ "$max_train_cameras" =~ ^[0-9]+$ ]]; then
        echo "오류: 최대 카메라 수는 0 이상의 정수여야 합니다."
        return 1
    fi

    read -p "품질 우선 비율(0-100, 기본: 70, 나머지는 랜덤): " quality_ratio_percent
    quality_ratio_percent=${quality_ratio_percent:-70}
    if ! [[ "$quality_ratio_percent" =~ ^[0-9]+$ ]] || [ "$quality_ratio_percent" -gt 100 ]; then
        echo "오류: 품질 우선 비율은 0~100 정수여야 합니다."
        return 1
    fi
    camera_quality_ratio=$("$PYTHON_BIN" - "$quality_ratio_percent" <<'PY'
import sys
print(int(sys.argv[1]) / 100.0)
PY
)

    read -p "카메라 랜덤 선택 시드(기본: 42): " camera_selection_seed
    camera_selection_seed=${camera_selection_seed:-42}
    if ! [[ "$camera_selection_seed" =~ ^[0-9]+$ ]]; then
        echo "오류: 시드는 0 이상의 정수여야 합니다."
        return 1
    fi

    scene_name=$(basename "$scene_path")
    IFS='_' read -r part1 part2 _ <<< "$scene_name"
    if [ -n "$part1" ] && [ -n "$part2" ]; then
        output_prefix="${part1}_${part2}"
    else
        output_prefix="$scene_name"
    fi

    if [ -f "$scene_path/sparse/0/images.bin" ]; then
        camera_count=$("$PYTHON_BIN" - "$scene_path/sparse/0/images.bin" <<'PY'
import sys
from utils.read_write_model import read_images_binary
print(len(read_images_binary(sys.argv[1])))
PY
)
    else
        camera_count=$(collect_image_files "$scene_path/input" | wc -l)
        echo "[WARN] sparse/0/images.bin 이 없어 input 이미지 수로 대체합니다: $camera_count"
    fi

    used_camera_count="$camera_count"
    if [ "$max_train_cameras" -gt 0 ] && [ "$max_train_cameras" -lt "$camera_count" ]; then
        used_camera_count="$max_train_cameras"
    fi
    model_path=$(next_available_output_dir "output/${output_prefix}_${used_camera_count}")

    echo ""
    echo "학습 시작... (완료될 때까지 기다려주세요)"
    echo "[INFO] COLMAP 카메라 수: $camera_count, 학습 사용 카메라 수: $used_camera_count"
    echo "[INFO] 선택 방식: 품질 ${quality_ratio_percent}% + 랜덤 $((100 - quality_ratio_percent))% (seed=$camera_selection_seed)"
    echo "[INFO] 출력 경로: $model_path"
    "$PYTHON_BIN" train.py -s "$scene_path" --iterations "$iterations" --max_train_cameras "$max_train_cameras" --camera_quality_ratio "$camera_quality_ratio" --camera_selection_seed "$camera_selection_seed" -m "$model_path"
    
    echo ""
    echo "✓ 학습 완료! 결과는 output/ 폴더에서 확인하세요"
}

# 함수: 렌더링
render_model() {
    echo ""
    echo "=== 렌더링 ==="
    echo ""
    echo "사용 가능한 모델:"
    ls -d output/*/ 2>/dev/null | nl
    
    echo ""
    echo "렌더링할 모델 폴더 경로를 입력하세요:"
    read -p "경로: " model_path
    
    if [ ! -d "$model_path" ]; then
        echo "오류: 모델 폴더를 찾을 수 없습니다: $model_path"
        return 1
    fi
    
    echo ""
    echo "렌더링 시작..."
    "$PYTHON_BIN" render.py -m "$model_path"
    
    echo ""
    echo "✓ 렌더링 완료!"
    echo "결과 위치: $model_path/train/renders/ 및 $model_path/test/renders/"
}

# 메인 메뉴
while true; do
    echo ""
    echo "================================"
    echo "  Gaussian Splatting 빠른 시작"
    echo "================================"
    echo "1) 데이터 준비 (이미지 → COLMAP)"
    echo "2) 동영상 -> 이미지 추출 (원본/사용자 화질)"
    echo "3) 모델 학습"
    echo "4) 렌더링"
    echo "5) 종료"
    echo ""
    read -p "선택 (1-5): " choice
    
    case $choice in
        1) prepare_data ;;
        2) extract_frames_from_video ;;
        3) train_model ;;
        4) render_model ;;
        5) echo "종료합니다."; exit 0 ;;
        *) echo "잘못된 선택입니다." ;;
    esac
done
