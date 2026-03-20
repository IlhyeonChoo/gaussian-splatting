#!/bin/bash
# Gaussian Splatting 환경 활성화 스크립트

cd /home/ilhyeonchu/ReCompose3D/3DGS/gaussian-splatting
source venv/bin/activate

echo ""
echo "✓ Gaussian Splatting 환경이 활성화되었습니다!"
echo ""
echo "사용 가능한 명령어:"
echo "  - python train.py -s <데이터_경로>     : 모델 학습"
echo "  - python render.py -m <모델_경로>      : 렌더링"
echo "  - python convert.py -s <이미지_폴더>   : 이미지를 COLMAP 데이터로 변환"
echo "  - ./quickstart.sh                      : 대화형 빠른 시작 메뉴"
echo ""
echo "자세한 가이드는 README_KR.md를 참조하세요"
echo ""

# 새로운 bash 세션 시작
bash
