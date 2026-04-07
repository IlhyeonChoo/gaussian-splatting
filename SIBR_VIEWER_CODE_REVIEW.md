# SIBR Viewer 코드 리뷰 및 아키텍처 분석

## 0. 범위와 관점

이 문서는 이 저장소 안의 SIBR viewer 계층 중에서 실제 3D Gaussian Splatting 사용 경로를 중심으로 분석한다.

- 공통 프레임워크: `SIBR_viewers/src/core/*`
- 로컬 Gaussian viewer: `SIBR_viewers/src/projects/gaussianviewer/*`
- 학습 연동 remote viewer: `SIBR_viewers/src/projects/remote/*`
- Python 연동부: `train.py`, `gaussian_renderer/network_gui.py`, `gaussian_renderer/__init__.py`, `scene/__init__.py`, `utils/camera_utils.py`

분석 기준은 다음 4가지다.

1. 아키텍처가 어떻게 계층화되어 있는가
2. 실행 시 어떤 흐름으로 데이터가 이동하고 렌더링되는가
3. 중요한 함수/클래스가 무엇을 입력받아 무엇을 출력하는가
4. 그 입력은 어디서 오고, 출력은 어디로 가서 어떻게 소비되는가

이 문서는 `basic`, `ulr` 같은 다른 SIBR 프로젝트 전체를 다루지 않는다. 다만 공통 코어 계층은 Gaussian viewer가 실제로 타는 경로만 추적한다.

---

## 1. 전체 아키텍처 구조

### Part 1. 레이어 구조

SIBR viewer는 대략 아래 레이어로 나뉜다.

| 레이어 | 주 역할 | 이 프로젝트에서 중요한 파일 |
| --- | --- | --- |
| `core/system` | CLI, 문자열, 행렬/벡터, 유틸 | `CommandLineArgs.*` |
| `core/graphics` | OpenGL window/texture/shader/render target | `Window.*`, `RenderTarget.*`, `Shader.*` |
| `core/assets` | 카메라/이미지/리소스 표현 | `InputCamera.*` |
| `core/scene` | dataset을 scene 객체로 조립 | `ParseData.*`, `BasicIBRScene.*`, `ProxyMesh.*`, `InputImages.*`, `CalibratedCameras.*` |
| `core/view` | view 인터페이스, camera handler, multi-view orchestration | `ViewBase.*`, `MultiViewManager.*`, `InteractiveCameraHandler.*`, `RenderingMode.*` |
| `core/renderer` | 일반 렌더 패스 | `PointBasedRenderer.*`, `CopyRenderer.*` |
| `projects/gaussianviewer` | 로컬 Gaussian PLY viewer | `apps/gaussianViewer/main.cpp`, `GaussianView.*`, `GaussianSurfaceRenderer.*` |
| `projects/remote` | 학습 프로세스와 TCP로 연결되는 viewer | `apps/remoteGaussianUI/main.cpp`, `RemotePointView.*` |

핵심 포인트는 다음이다.

- `core/*`는 재사용 가능한 프레임워크다.
- 실제 앱 엔트리포인트는 `projects/*/apps/*/main.cpp`에 있다.
- Gaussian viewer는 `ViewBase` 기반 뷰 하나를 만들어 `MultiViewManager`에 등록한다.
- `MultiViewManager`는 입력 처리, 카메라 업데이트, 오프스크린 렌더, GUI 표시를 담당한다.
- 실제 Gaussian splat 렌더링은 `GaussianView::onRenderIBR()` 안에서 일어난다.

### Part 2. 이 저장소에서 중요한 두 가지 viewer

#### 1) 로컬 viewer: `gaussianViewer`

목적:

- 학습 결과 폴더(`model_path`)에 저장된 `point_cloud/iteration_*/point_cloud.ply`를 직접 읽어 렌더링
- `cameras.json`, `input.ply`를 이용해 camera/top view/proxy mesh도 함께 구성

특징:

- Python 프로세스 없이 단독 실행 가능
- CUDA rasterizer를 C++에서 직접 호출
- 주 렌더 모드는 `Splats`

#### 2) 원격 viewer: `remoteGaussianUI`

목적:

- 학습 중인 Python 프로세스에 카메라를 보내고
- Python이 현재 Gaussian 모델로 렌더링한 이미지를 받아 표시

특징:

- viewer 자체는 최종 Gaussian 이미지를 계산하지 않는다
- TCP JSON + raw RGB 바이트 프로토콜을 사용한다
- 응답이 늦는 동안엔 proxy/SfM point fallback을 보여줄 수 있다

---

## 2. SIBR viewer가 읽는 입력은 어디서 만들어지는가

### Part 3. Python 학습 코드가 viewer 입력 파일을 만들어 놓는다

로컬 Gaussian viewer가 `model_path`만으로 동작하는 이유는 Python 쪽이 필요한 입력을 미리 만들어 두기 때문이다.

#### `scene/__init__.py::Scene.__init__`

입력:

- `args.source_path`
- `args.model_path`
- train/test camera info
- source point cloud

출력:

- `<model_path>/input.ply`
- `<model_path>/cameras.json`

출력 사용처:

- `SIBR_viewers/src/core/scene/ParseData.cpp::getParsedGaussianData`
- `SIBR_viewers/src/core/assets/InputCamera.cpp::loadJSON`

구체적으로:

1. source dataset의 point cloud를 `input.ply`로 복사한다.
2. train/test camera를 `camera_to_JSON()`으로 직렬화해 `cameras.json`으로 저장한다.

#### `utils/camera_utils.py::camera_to_JSON`

입력:

- Python camera info 객체(`CameraInfo`와 같은 `R/T/Fov/width/height/image_name` 보유 객체)

출력 필드:

- `id`
- `img_name`
- `width`
- `height`
- `position`
- `rotation`
- `fy`
- `fx`

출력 사용처:

- `InputCamera::loadJSON()`

즉, SIBR viewer는 자체적으로 COLMAP를 다시 읽는 것이 아니라, Python 쪽이 정리해 준 `cameras.json`을 읽어 C++ `InputCamera` 벡터로 복구한다.

#### `scene/__init__.py::Scene.save`

입력:

- 현재 iteration의 `GaussianModel`

출력:

- `<model_path>/point_cloud/iteration_<N>/point_cloud.ply`

출력 사용처:

- `gaussianViewer/main.cpp`
- `GaussianView::GaussianView()`

즉, 로컬 Gaussian viewer의 핵심 입력은 아래 3개다.

- `cfg_args`
- `cameras.json`
- `point_cloud/iteration_*/point_cloud.ply`

---

## 3. 공통 viewer 프레임워크가 어떻게 동작하는가

### Part 4. 앱 공통 실행 프레임

로컬 viewer와 remote viewer는 둘 다 아래 구조를 공유한다.

```text
main.cpp
  -> CommandLineArgs 파싱
  -> Window 생성
  -> BasicIBRScene 생성
  -> ViewBase 파생 뷰 생성
  -> InteractiveCameraHandler 생성
  -> MultiViewManager에 뷰 등록
  -> while(window.isOpened()):
       Input::poll()
       MultiViewManager.onUpdate()
       MultiViewManager.onRender()
```

### Part 5. `MultiViewManager` 기준 호출 체인

중요 호출 체인은 아래와 같다.

```text
Input::poll()
  -> MultiViewManager::onUpdate()
    -> 각 IBR subview의 handler->update(...)
    -> handler가 현재 eye camera를 갱신
  -> MultiViewManager::onRender()
    -> renderSubView(...)
      -> IBRSubView::render(...)
        -> IRenderingMode::render(...)
          -> ViewBase::onRenderIBR(...)
```

실제 파일 기준으로 보면:

- `MultiViewBase::onUpdate()`는 subview별 입력을 자르고 camera handler를 갱신한다.
- `MultiViewBase::IBRSubView::render()`는 현재 rendering mode에 렌더를 위임한다.
- 기본 모드인 `MonoRdrMode::render()`는 `view.onRenderIBR(*_destRT, eye)`를 호출한다.
- 즉, Gaussian viewer에서 실제 최종 이미지를 만드는 함수는 `GaussianView::onRenderIBR()`다.

### Part 6. 카메라 입력은 어떻게 eye camera가 되는가

`InteractiveCameraHandler`의 역할:

- 키보드/마우스 입력을 받아 FPS/orbit/trackball/interpolation 카메라를 갱신
- 현재 카메라를 `getCamera()`로 제공
- 필요하면 smoothing, snap, camera recorder까지 처리

입력:

- `Input`
- `deltaTime`
- `Viewport`
- 초기 기준 camera set

출력:

- 현재 `InputCamera _currentCamera`

출력 사용처:

- `MultiViewBase::onUpdate()`에서 IBR subview의 `cam`
- 이후 `MonoRdrMode::render()`가 이 카메라를 `onRenderIBR(dst, eye)`로 넘김

즉, view 자신은 사용자 입력을 직접 처리하지 않아도 된다. 실제 camera state는 외부의 handler가 관리하고, 뷰는 렌더링만 한다.

---

## 4. scene 조립 단계: dataset path가 camera/image/proxy로 바뀌는 과정

### Part 7. `BasicIBRScene`는 scene assembler다

`BasicIBRScene`의 핵심 역할은 `ParseData`를 실제 scene 컴포넌트로 조립하는 것이다.

입력:

- `BasicIBRAppArgs`
- `SceneOptions`

중간 산출물:

- `_data`: `ParseData`
- `_cams`: `CalibratedCameras`
- `_imgs`: `InputImages`
- `_proxies`: `ProxyMesh`
- `_renderTargets`: `RenderTargetTextures`

출력 사용처:

- `GaussianView`
- `RemotePointView`
- `SceneDebugView`

### Part 8. `ParseData`가 하는 일

`ParseData::getParsedData()`는 dataset path를 보고 어떤 형식인지 판별하고, 그에 맞는 파서를 부른다.

Gaussian 관련 경로는 사실상 아래 두 가지다.

1. auto-detect로 `cameras.json`을 발견하는 경우
2. 사용자가 명시적으로 `dataset_type=gaussian`을 지정하는 경우

정상 경로라면 `getParsedGaussianData()`가 호출되고, 그 안에서:

- `InputCamera::loadJSON(dataset_path + "/cameras.json")`
- `_meshPath = dataset_path + "/input.ply"`
- `_basePathName = dataset_path`
- `_imgPath = "."`

를 세팅한다.

이후 `BasicIBRScene::createFromData()`가:

- `CalibratedCameras::setupFromData(_data)`로 `_camInfos`를 그대로 `_inputCameras`로 넘기고
- `InputImages::loadFromData(_data)`로 image를 읽고
- `ProxyMesh::loadFromData(_data)`로 `input.ply`를 proxy mesh로 로드한다.

### Part 9. scene 조립의 input/output 추적

#### `InputCamera::loadJSON`

입력:

- `cameras.json` 배열

읽는 필드:

- `id`
- `img_name`
- `width`
- `height`
- `fy`
- `fx`
- `position`
- `rotation`

가공:

- `InputCamera(fy, fx, ...)` 생성
- orientation의 2, 3번째 축(`col(1)`, `col(2)`)에 부호 반전 적용

출력:

- `std::vector<InputCamera::Ptr>`

출력 사용처:

- `ParseData::_camInfos`
- `BasicIBRScene::_cams`
- `InteractiveCameraHandler::setup()`

#### `InputImages::loadFromData`

입력:

- `data->imgInfos()`
- `data->imgPath()`
- `data->activeImages()`

출력:

- `_inputImages`

출력 사용처:

- top view/scene overview
- 일부 디버그 뷰

중요한 점:

- Gaussian splat 본 렌더링 자체는 input image 없이도 가능하다.
- 그래서 `gaussianViewer/main.cpp`에서는 `myOpts.images = myArgs.loadImages`로 optional하게만 켠다.

#### `ProxyMesh::loadFromData`

입력:

- `data->meshPath()` 보통 `input.ply`

출력:

- `_proxy` (`Mesh`)

출력 사용처:

- `Raycaster::addMesh(scene->proxies()->proxy())`
- `PointBasedRenderer` fallback
- `SceneDebugView` top view

즉, proxy mesh는 Gaussian splat 그 자체를 그리기 위해 쓰는 것이 아니라:

- top view
- raycast 기반 camera interaction
- fallback point rendering

쪽에 더 가깝다.

---

## 5. 로컬 Gaussian viewer(`gaussianViewer`)의 상세 동작

### Part 10. 엔트리포인트에서 scene을 띄우는 과정

파일: `SIBR_viewers/src/projects/gaussianviewer/apps/gaussianViewer/main.cpp`

초기화 흐름:

1. `GaussianAppArgs` 파싱
2. `--model-path`와 `-m`, `--path`와 `-s`를 정규화
3. `<model_path>/cfg_args`를 읽어서:
   - `source_path`
   - `sh_degree`
   - `white_background`
   를 문자열 파싱으로 추출
4. `BasicIBRScene` 생성
5. `<model_path>/point_cloud/iteration_*/point_cloud.ply` 경로 결정
6. `GaussianView` 생성
7. `Raycaster`, `InteractiveCameraHandler`, `MultiViewManager` 구성
8. `"Point view"`와 `"Top view"` 등록
9. 메인 루프 진입

### Part 11. `main.cpp`의 중요한 input/output

#### `cfg_args` 파싱

입력:

- `<model_path>/cfg_args`

입력 출처:

- `train.py::prepare_output_and_logger()`

출력:

- `dataset_path`
- `sh_degree`
- `white_background`

출력 사용처:

- `BasicIBRScene(myArgs, myOpts)`
- `GaussianView(..., sh_degree, white_background, ...)`

#### point cloud 경로 선택

입력:

- `<model_path>/point_cloud/iteration_*`
- 선택적 CLI `--iteration`

출력:

- 실제 PLY 파일 경로 문자열

출력 사용처:

- `GaussianView` 생성자

기본 동작은 가장 큰 iteration 번호를 찾아 마지막 저장본을 연다.

### Part 12. `GaussianView` 생성자에서 하는 일

파일: `SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp`

입력:

- `BasicIBRScene::Ptr`
- render width/height
- PLY 파일 경로
- `sh_degree`
- `white_background`
- `useInterop`
- `device`

출력:

- CPU SoA arrays
- CUDA device arrays
- GL/CUDA interop output buffer
- fallback CPU roundtrip buffer
- 진단용 `GaussianData`

출력 사용처:

- `onRenderIBR()`
- `Ellipsoids` 모드
- crop/save UI

#### 내부 처리 순서

1. CUDA device 확인 및 선택
2. `PointBasedRenderer`와 `BufferCopyRenderer` 생성
3. active camera를 debug view용으로 표시
4. `loadPly<D>()`로 Gaussian 데이터 로드
5. CPU 벡터를 CUDA 메모리로 복사
6. 배경색, view/proj/cam_pos용 CUDA 버퍼 생성
7. `GaussianData`와 `GaussianSurfaceRenderer` 생성
8. 최종 RGB 결과를 받을 GL buffer 생성
9. 가능하면 CUDA-OpenGL interop 등록
10. 실패하면 CPU roundtrip fallback 준비

### Part 13. `loadPly<D>()`가 실제로 하는 데이터 변환

입력:

- binary PLY 파일

입력 출처:

- `Scene.save()`가 저장한 `<model_path>/point_cloud/iteration_N/point_cloud.ply`

가정하는 레이아웃:

- position
- normal
- SH coefficients
- opacity
- scale
- rotation

출력:

- `pos`
- `shs`
- `opacities`
- `scales`
- `rot`
- `minn/maxx`

출력 사용처:

- CUDA 업로드
- crop box 범위 초기화
- `savePly()` 재저장

이 함수의 중요한 변환:

1. AoS(`RichPoint<D>`)로 한 번에 읽는다.
2. bbox를 구한 뒤 Morton order로 정렬한다.
3. SoA로 재배열한다.
4. quaternion을 정규화한다.
5. scale은 `exp()`로 복원한다.
6. opacity는 `sigmoid()`로 복원한다.
7. SH coefficient는 rasterizer가 기대하는 채널 순서로 재배치한다.

즉, viewer는 PLY를 그대로 GPU에 올리지 않는다. 학습 저장 포맷을 viewer용 SoA 포맷으로 바꿔서 올린다.

### Part 14. 한 프레임이 렌더링되는 실제 흐름

로컬 Gaussian viewer의 핵심 프레임 흐름은 아래와 같다.

```text
window loop
  -> Input::poll()
  -> MultiViewManager::onUpdate()
    -> InteractiveCameraHandler::update()
    -> 현재 eye camera 결정
  -> MultiViewManager::onRender()
    -> MonoRdrMode::render()
      -> GaussianView::onRenderIBR(dst, eye)
        -> CudaRasterizer::Rasterizer::forward(...)
        -> BufferCopyRenderer::process(imageBuffer, dst)
      -> screen quad로 window에 blit
```

### Part 15. `GaussianView::onRenderIBR()`의 세 가지 모드

#### 1) `Splats`

가장 중요한 실제 렌더링 경로다.

입력:

- `dst`: 현재 render target
- `eye`: 현재 novel view camera
- 내부 CUDA 버퍼:
  - `pos_cuda`
  - `shs_cuda`
  - `opacity_cuda`
  - `scale_cuda`
  - `rot_cuda`
  - `background_cuda`

중간 처리:

1. SIBR camera 행렬을 rasterizer 좌표계에 맞게 보정
   - `view`의 row 1, 2 부호 반전
   - `viewproj`의 row 1 부호 반전
2. `tan_fovx`, `tan_fovy` 계산
3. 현재 카메라 행렬과 camera position을 CUDA 메모리에 복사
4. interop 가능하면 GL buffer를 CUDA 포인터로 매핑
5. `CudaRasterizer::Rasterizer::forward(...)` 호출
6. interop이면 unmap, 아니면 CPU로 복사 후 GL buffer에 업로드
7. `BufferCopyRenderer`가 GL buffer를 `dst`에 복사

출력:

- planar float RGB image buffer
- 최종적으로 `dst`에 기록된 color image

출력 사용처:

- `MonoRdrMode::render()`가 screen quad로 window 또는 상위 RT에 표시

#### 2) `Initial Points`

입력:

- `scene->proxies()->proxy()`
- `eye`
- `dst`

출력:

- proxy mesh point rendering

출력 사용처:

- 디버그/초기 SfM-like visualization

중요:

- 이 모드는 Gaussian point cloud 자체를 그리는 것이 아니라 proxy mesh의 points를 그린다.

#### 3) `Ellipsoids`

입력:

- `GaussianData`
- `eye`
- `dst`

출력:

- ellipsoid surface diagnostic image

출력 사용처:

- 진단/디버그 모드

### Part 16. `CudaRasterizer::Rasterizer::forward()`로 넘어가는 실제 입력

`GaussianView`가 넘기는 입력은 거의 Python rasterizer와 동일한 의미를 가진다.

입력:

- `P`: Gaussian 개수
- `D`: SH degree
- `M`: 최대 SH coeff count(여기서는 16)
- `background`
- `width`, `height`
- `means3D`
- `shs`
- `opacities`
- `scales`
- `scale_modifier`
- `rotations`
- `viewmatrix`
- `projmatrix`
- `cam_pos`
- `tan_fovx`, `tan_fovy`
- `out_color`
- `antialiasing`
- optional `radii`

출력:

- `out_color`
- 내부 geometry/binning/image working buffers
- optional `radii`

출력 사용처:

- `BufferCopyRenderer`
- crop/culling bookkeeping

`forward.cu` 안에서 일어나는 핵심 처리:

1. frustum culling
2. scale/rotation -> 3D covariance
3. 3D covariance -> 2D screen covariance
4. SH -> RGB 변환
5. Gaussian별 screen tile overlap 계산
6. tile-depth key로 정렬
7. tile 단위 splat accumulation
8. 최종 RGB planar buffer 작성

즉, 로컬 SIBR viewer는 Python과 같은 rasterizer 패밀리를 쓰되, PyTorch extension이 아니라 C++/CUDA API를 직접 호출한다.

### Part 17. `BufferCopyRenderer`는 왜 필요한가

`CudaRasterizer::forward()`의 출력은 texture가 아니라 GL shader storage buffer에 기록된 planar float 배열이다.

`BufferCopyRenderer::process()`는:

- 입력: `bufferID`, `dst`, `width`, `height`
- 내부 shader: `projects/gaussianviewer/renderer/shaders/copy.frag`
- 출력: `dst`에 full-screen quad 렌더

shader는:

- SSBO binding 0에서 `R plane`, `G plane`, `B plane`을 읽고
- 화면 좌표에 맞춰 샘플링해
- `vec4(r, g, b, 1)`로 출력한다.

즉, CUDA 결과를 화면에 보이는 일반 color RT로 바꾸는 마지막 브리지다.

### Part 18. Ellipsoid 모드는 어떻게 렌더링되는가

`GaussianSurfaceRenderer`는 main splat path와 별도다.

입력:

- `GaussianData`: mean/rotation/scale/alpha/color의 GL buffer 집합
- `Camera eye`
- `IRenderTarget dst`

출력:

- 내부 FBO color attachment
- 최종적으로 `dst`에 blit된 color image

작동 방식:

1. Gaussian 하나당 box instance 하나를 그림
2. vertex shader가 quaternion/scale/center를 읽어 ellipsoid bounding geometry를 구성
3. fragment shader가 ray-ellipsoid intersection을 계산
4. stage 0은 solid pass, stage 1은 additive blended halo 비슷한 pass
5. 결과를 `dst`로 blit

이 경로는 Gaussian splatting의 본 경로가 아니라 “ellipsoid surface를 대충 시각화하는 디버그 경로”에 가깝다.

### Part 19. crop/save 기능의 데이터 흐름

GUI에서 crop box를 켜고 Save를 누르면:

1. CUDA device arrays를 다시 CPU로 `cudaMemcpy`
2. `savePly()`가 현재 bbox 안에 들어오는 Gaussian만 필터링
3. scale은 `log()`, opacity는 `inverse_sigmoid()`로 저장 포맷에 맞게 되돌림
4. 새 binary PLY를 씀

즉, 저장되는 PLY는 “현재 메모리 안의 viewer state”를 다시 학습 포맷에 가깝게 역직렬화한 결과다.

---

## 6. remote viewer(`remoteGaussianUI`)와 Python 학습 루프의 연동

### Part 20. remote viewer 전체 흐름

remote path는 아래와 같다.

```text
RemotePointView (C++)
  -> JSON request 전송
  -> Python network_gui.receive()
  -> MiniCam 생성
  -> gaussian_renderer.render()
  -> RGB bytes 응답
  -> RemotePointView가 texture 업데이트
  -> CopyRenderer로 화면 표시
```

### Part 21. viewer 쪽 엔트리포인트

파일: `SIBR_viewers/src/projects/remote/apps/remoteGaussianUI/main.cpp`

초기화 흐름:

1. `RemoteAppArgs` 파싱
2. `Window` 생성
3. `MultiViewManager` 생성
4. `RemotePointView(ip, port)` 생성
5. path override가 있으면 즉시 `resetScene()`
6. loop 중 `remoteView->sceneName()`가 바뀌면 `dataset_path`를 갱신하고 `resetScene()` 재실행
7. 매 프레임 `multiViewManager.onUpdate/onRender`

즉, remote viewer는 “렌더링 이미지는 원격에서 받지만, local scene metadata와 top view는 필요할 때 다시 로드하는 구조”다.

### Part 22. `resetScene()`의 역할

입력:

- `myArgs.dataset_path`
- render size
- 기존 `RemotePointView`

출력:

- 새 `BasicIBRScene`
- 새 camera handler
- 새 `Top view`
- 재연결된 `Point view`

출력 사용처:

- `RemotePointView::setScene()`
- `SceneDebugView`
- `MultiViewManager`

즉, remote viewer는 network image만 바꾸는 것이 아니라, scene path가 바뀌면 camera/proxy/top view 전체를 다시 조립한다.

### Part 23. `RemotePointView::send_receive()`의 프로토콜

입력:

- `_remoteInfo`
  - resolution
  - fovy/fovx
  - znear/zfar
  - view matrix
  - viewProj matrix
- GUI 상태
  - training on/off
  - SH Python on/off
  - Rot/Scale Python on/off
  - keep_alive
  - scaling_modifier

출력:

- TCP JSON request
- 수신 image bytes
- 수신 scene name string

출력 사용처:

- `_imageData` -> `onRenderIBR()`
- `current_scene` -> `main.cpp`의 `resetScene()`

### Part 24. viewer -> Python으로 넘어가는 필드

JSON request 필드:

- `resolution_x`
- `resolution_y`
- `fov_y`
- `fov_x`
- `z_far`
- `z_near`
- `train`
- `view_matrix`
- `view_projection_matrix`
- `scaling_modifier`
- `shs_python`
- `rot_scale_python`
- `keep_alive`

이 필드들은 Python에서 `network_gui.receive()`가 읽는다.

### Part 25. Python 쪽 수신 처리

파일: `gaussian_renderer/network_gui.py`

#### `receive()`

입력:

- socket으로 받은 JSON

출력:

- `MiniCam`
- `do_training`
- `do_shs_python`
- `do_rot_scale_python`
- `keep_alive`
- `scaling_modifier`

출력 사용처:

- `train.py`의 학습 루프

중요한 처리:

- C++에서 온 `view_matrix`, `view_projection_matrix`를 tensor로 복원
- `view_matrix`는 column 1, 2의 부호를 뒤집고, `view_projection_matrix`는 column 1의 부호를 뒤집어 Python/CUDA rasterizer 좌표계로 맞춘다
- `MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)` 생성

즉, remote viewer는 카메라를 “이미 완성된 view/projection matrix 형태”로 넘기고, Python은 그것을 `MiniCam`으로 감싼 뒤 바로 렌더러에 투입한다.

### Part 26. Python 렌더러가 무엇을 입력받아 무엇을 내보내는가

파일: `gaussian_renderer/__init__.py::render`

입력:

- `viewpoint_camera` (`Camera` 또는 `MiniCam`)
- `GaussianModel pc`
- `pipe`
- `bg_color`
- `scaling_modifier`

출력:

- `render`: RGB tensor
- `viewspace_points`
- `visibility_filter`
- `radii`
- `depth`

출력 사용처:

- remote viewer 응답 이미지
- 학습 loss 계산
- densification/pruning statistics

remote path에서는 이 중 `render`만 바로 네트워크로 전송한다.

### Part 27. train.py 안에서 remote viewer 요청이 처리되는 방식

`train.py`의 training loop는 매 iteration마다:

1. `network_gui.try_connect()`로 viewer 연결을 확인
2. 연결되어 있으면 반복적으로:
   - `network_gui.receive()`
   - `render(custom_cam, gaussians, pipe, background, scaling_modifier=...)`
   - 결과 RGB tensor를 `uint8` bytes로 변환
   - `network_gui.send(net_image_bytes, dataset.source_path)`
3. viewer가 `train=true`를 보내면 학습 루프로 복귀

여기서 `dataset.source_path`는 image 뒤에 붙는 scene name 문자열이며, C++ remote viewer가 `current_scene`로 받아 scene reload 판단에 쓴다.

### Part 28. remote viewer 화면 표시 경로

`RemotePointView::onRenderIBR()`는 아래처럼 동작한다.

1. camera/resolution이 바뀌면 `_timestampRequested++`
2. 아직 최신 image가 안 왔으면 `preview=true`
3. 다음 조건이면 proxy point fallback:
   - `_showSfM`
   - 아직 이미지 없음
   - motion 중 preview인데 `_renderSfMInMotion`
4. 그렇지 않으면 `_imageData`를 GL texture에 업로드
5. `CopyRenderer`로 `dst`에 출력

즉, remote viewer는 “항상 최신 이미지가 올 때까지 block”하지 않는다. 상황에 따라 proxy mesh point preview를 보여 준다.

---

## 7. 중요 포인트별 Input -> Output 추적 요약

### Part 29. 핵심 함수별 입출력 맵

#### `scene/__init__.py::Scene.__init__`

- Input: source dataset, camera info, point cloud
- Input source: Python dataset loader
- Output: `input.ply`, `cameras.json`
- Output sink: SIBR local viewer의 Gaussian dataset bootstrap

#### `utils/camera_utils.py::camera_to_JSON`

- Input: Python camera info 객체(`R/T/Fov/width/height/image_name` 보유)
- Input source: dataset loader가 만든 camera objects
- Output: `id/img_name/width/height/position/rotation/fx/fy`
- Output sink: `InputCamera::loadJSON`

#### `ParseData::getParsedGaussianData`

- Input: `dataset_path`
- Input source: `gaussianViewer`의 `model_path` 또는 `source_path`
- Output: `_camInfos`, `_meshPath`, `_imgPath`, `_basePathName`
- Output sink: `BasicIBRScene::createFromData`

#### `InputCamera::loadJSON`

- Input: `cameras.json`
- Input source: Python `Scene.__init__`
- Output: `vector<InputCamera>`
- Output sink: `CalibratedCameras`, `InteractiveCameraHandler`

#### `BasicIBRScene::createFromData`

- Input: `ParseData`
- Input source: dataset parsing
- Output: cameras/images/proxy/render targets
- Output sink: `GaussianView`, `RemotePointView`, `SceneDebugView`

#### `GaussianView::GaussianView`

- Input: scene, resolution, PLY path, SH degree, background
- Input source: `main.cpp`
- Output: CUDA buffers, GL interop buffer, debug renderer state
- Output sink: `GaussianView::onRenderIBR`

#### `GaussianView::onRenderIBR`

- Input: `dst`, `eye`
- Input source: `MonoRdrMode::render`, `InteractiveCameraHandler`
- Output: `dst` color image
- Output sink: `MonoRdrMode`의 화면 blit

#### `CudaRasterizer::Rasterizer::forward`

- Input: Gaussian arrays + camera matrices + output pointer
- Input source: `GaussianView`
- Output: RGB float image buffer
- Output sink: `BufferCopyRenderer`

#### `RemotePointView::send_receive`

- Input: view/proj/resolution/train flags
- Input source: remote viewer GUI + camera handler
- Output: JSON request / image bytes / scene path
- Output sink: Python `network_gui`, local `onRenderIBR`, `resetScene`

#### `network_gui.receive`

- Input: viewer JSON request
- Input source: `RemotePointView`
- Output: `MiniCam` + flags
- Output sink: `train.py`

#### `gaussian_renderer.render`

- Input: `MiniCam`, `GaussianModel`, pipeline state
- Input source: training loop
- Output: rendered RGB/depth/radii
- Output sink: remote viewer response + 학습 loss/densification

---

## 8. 코드 리뷰 관점에서 본 중요한 포인트

### Part 30. 중요한 설계 특징

이 구현에서 특히 중요한 설계 선택은 다음 3가지다.

1. **SIBR viewer는 scene metadata와 Gaussian payload를 분리한다.**
   - camera/top view/proxy는 `BasicIBRScene`
   - 실제 Gaussian splat image는 `GaussianView`

2. **로컬 viewer와 Python 학습 렌더러는 같은 의미의 rasterization 파라미터를 공유한다.**
   - position / scale / rotation / opacity / SH
   - view/projection
   - scale modifier
   - antialiasing

3. **view 입력과 렌더링 구현을 분리한다.**
   - 입력/카메라: `InteractiveCameraHandler`
   - orchestration: `MultiViewManager`
   - 실제 렌더: `ViewBase::onRenderIBR`

이 덕분에 `GaussianView`는 camera UI 코드를 몰라도 되고, `RemotePointView`도 같은 orchestration 위에 올라갈 수 있다.

### Part 31. 리뷰 findings

아래는 실제 코드 기준으로 눈에 띄는 이슈들이다.

#### High 1. `dataset_type=gaussian` 명시 분기가 잘못되어 있다

파일:

- `SIBR_viewers/src/core/scene/ParseData.cpp`

문제:

- `datasetTypeStr == "gaussian"`일 때 `_datasetType = Type::BLENDER`로 설정되어 있다.
- 따라서 사용자가 `--dataset_type gaussian`을 명시하면 의도와 다르게 Blender 로더가 선택된다.

영향:

- `cameras.json` 기반 scene을 명시적으로 열려는 경우 실패 가능
- auto-detect는 `else if (fileExists(gaussian)) _datasetType = Type::GAUSSIAN;`라서 우연히 정상 동작할 수 있음

권장 수정:

- 해당 분기를 `Type::GAUSSIAN`으로 바꾸고 에러 메시지의 변수도 `blender`가 아니라 `gaussian` 경로를 쓰도록 수정

#### High 2. Gaussian dataset의 image path 설정이 취약하다

파일:

- `SIBR_viewers/src/core/scene/ParseData.cpp`

문제:

- `getParsedGaussianData()`가 `_imgPath = "."`로 고정한다.
- 그런데 `cameras.json`의 `img_name`은 Python `camera_to_JSON()`에서 보통 파일명만 저장한다.

영향:

- `--load_images`를 켜면 `InputImages::loadFromData()`가 `./<img_name>`를 읽으려 해서 현재 작업 디렉터리에 의존하게 된다.
- `model_path`를 dataset root 밖에서 열면 이미지 로딩이 깨질 가능성이 높다.

권장 수정:

- `_imgPath = dataset_path` 또는 실제 source image directory를 명시적으로 저장하도록 바꾸는 편이 안전

#### High 3. remote Python 프로토콜의 `recv()` 사용이 부분 수신에 취약하다

파일:

- `gaussian_renderer/network_gui.py`

문제:

- `read()`에서 `conn.recv(4)`와 `conn.recv(messageLength)`를 한 번씩만 호출한다.
- TCP는 요청 크기만큼 한 번에 다 주지 않을 수 있다.

영향:

- 큰 JSON이나 느린 네트워크 환경에서 message body가 잘리는 잠재 버그

권장 수정:

- 정확한 길이를 다 읽을 때까지 loop를 도는 `recv_exact()` helper를 두는 것이 안전

#### Medium 1. `RemotePointView::current_scene`는 스레드 동기화 없이 읽고 쓴다

파일:

- `SIBR_viewers/src/projects/remote/renderer/RemotePointView.cpp`

문제:

- network thread가 `current_scene`를 쓰고
- main thread가 `sceneName()`을 통해 읽는데 mutex가 없다.

영향:

- `std::string`에 대한 data race 가능

권장 수정:

- `current_scene` 접근을 mutex로 보호하거나 atomic-like handoff 구조로 변경

#### Medium 2. `GaussianView`는 일부 heap 자원을 해제하지 않는다

파일:

- `SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp`

문제:

- `gData = new GaussianData(...)`
- `_gaussianRenderer = new GaussianSurfaceRenderer()`
- destructor에서 둘 다 `delete`하지 않는다.

영향:

- viewer 종료 시 메모리/GL 리소스 누수

권장 수정:

- `std::unique_ptr`로 바꾸거나 destructor에서 정리

#### Medium 3. `GaussianSurfaceRenderer::process()`의 resize 시점이 어색하다

파일:

- `SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianSurfaceRenderer.cpp`

문제:

- FBO bind 후 먼저 clear를 하고, target size mismatch가 있으면 그 뒤에 `makeFBO()`를 호출한다.

영향:

- 리사이즈 직후 새 attachment가 명시적으로 clear되지 않은 채 사용될 수 있다.

권장 수정:

- size check 및 `makeFBO()`를 clear보다 먼저 수행

#### Medium 4. `BasicIBRScene` 생성자의 `BasicIBRScene();` 호출은 위임 생성이 아니다

파일:

- `SIBR_viewers/src/core/scene/BasicIBRScene.cpp`

문제:

- 생성자 안에서 `BasicIBRScene();`를 일반 함수처럼 호출한다.
- C++11 delegating constructor가 아니라 임시 객체 생성이라 실제 member init에는 쓰이지 않는다.

영향:

- 지금 코드에서는 이후 member reset으로 대부분 덮이지만, 읽는 사람에게 매우 혼란스럽고 잠재 버그 포인트다.

권장 수정:

- delegating constructor 문법으로 바꾸거나 공통 초기화 함수를 분리

#### Low 1. `cfg_args` 파싱은 문자열 포맷에 강하게 결합되어 있다

파일:

- `SIBR_viewers/src/projects/gaussianviewer/apps/gaussianViewer/main.cpp`

문제:

- `findArg()`가 Python `Namespace(...)` 문자열을 직접 substring 파싱한다.

영향:

- Python 쪽 포맷이 조금만 바뀌어도 viewer bootstrap이 깨질 수 있다.

권장 수정:

- JSON 또는 별도 config 파일 형식으로 저장

#### Low 2. PLY 로더는 저장 포맷에 매우 강하게 결합되어 있다

파일:

- `SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp`

문제:

- `loadPly<D>()`가 exact binary layout과 SH property ordering을 강하게 가정한다.

영향:

- 저장 포맷이 조금만 달라져도 viewer가 바로 깨진다.

권장 수정:

- header 기반 property 매핑 또는 포맷 버전 명시

---

## 9. 결론

### Part 32. 한 문장 요약

이 저장소의 SIBR viewer는 다음처럼 이해하면 가장 정확하다.

- `BasicIBRScene`가 **camera / image / proxy mesh 메타데이터 레이어**
- `InteractiveCameraHandler + MultiViewManager`가 **입력/뷰 orchestration 레이어**
- `GaussianView` 또는 `RemotePointView`가 **실제 이미지 생성 레이어**

### Part 33. 가장 중요한 런타임 경로

로컬 viewer:

```text
model_path
  -> cfg_args / cameras.json / input.ply / point_cloud.ply
  -> BasicIBRScene + GaussianView
  -> MultiViewManager + InteractiveCameraHandler
  -> GaussianView::onRenderIBR
  -> CudaRasterizer::forward
  -> BufferCopyRenderer
  -> window
```

remote viewer:

```text
camera input
  -> RemotePointView JSON request
  -> network_gui.receive
  -> MiniCam
  -> gaussian_renderer.render
  -> RGB bytes
  -> RemotePointView texture upload
  -> window
```

### Part 34. 실무적으로 봐야 할 포인트

이 코드베이스를 수정하거나 확장할 때 가장 먼저 봐야 하는 축은 아래 4개다.

1. `scene/__init__.py`와 `camera_to_JSON()`
   - viewer 입력 파일이 여기서 만들어진다.

2. `ParseData`와 `BasicIBRScene`
   - dataset path가 실제 C++ scene 객체로 바뀌는 지점이다.

3. `MultiViewManager`와 `InteractiveCameraHandler`
   - 렌더러가 어떤 camera를 언제 받는지 결정된다.

4. `GaussianView::onRenderIBR()` 또는 `RemotePointView::send_receive()`
   - 최종 이미지가 만들어지는 실제 핵심 경로다.

---

## 10. 추가 심화 리뷰: PLY 파싱과 메모리 적재

### Part 35. `loadPly<D>()`는 사실상 “고정 포맷 binary struct reader”다

파일:

- `SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp`

이 함수는 일반적인 의미의 robust PLY parser가 아니다. 실제 동작은 다음에 가깝다.

1. 헤더에서 `element vertex <count>`가 3번째 의미 있는 줄에 있다고 가정
2. `end_header`까지 읽고 버림
3. 나머지 바디를 `std::vector<RichPoint<D>>`로 한 번에 읽음

즉, 이 함수는 “PLY라는 컨테이너 형식”을 읽는 것이 아니라, `RichPoint<D>`와 정확히 같은 바이너리 레이아웃을 가진 학습 결과 파일을 읽는다.

### Part 36. 메모리 관점에서 실제 적재 순서

`GaussianView` 생성자 기준 실제 적재 순서는 아래와 같다.

```text
point_cloud.ply
  -> ifstream
  -> vector<RichPoint<D>> points        (AoS, CPU)
  -> vector<Pos> / vector<SHs<3>> / ... (SoA, CPU)
  -> cudaMalloc + cudaMemcpy            (SoA, GPU)
  -> GaussianData                       (GL buffer 복제본)
  -> imageBuffer                        (최종 RGB용 GL buffer)
  -> geom/binning/img scratch buffers   (CUDA 임시 버퍼, lazy grow)
```

즉, 한 번 읽고 끝나는 구조가 아니라 같은 의미의 데이터가 아래처럼 여러 번 존재한다.

- 원본 AoS CPU 버퍼
- 변환된 SoA CPU 버퍼
- 영속 SoA CUDA 버퍼
- Ellipsoids 모드용 GL buffer 복제본

이 점은 코드 이해에 중요하다. `Splats` 모드는 CUDA 버퍼를 쓰고, `Ellipsoids` 모드는 `GaussianData`의 GL buffer를 쓴다.

### Part 37. `RichPoint<D>` -> viewer SoA로 재배열되는 이유

입력 struct:

```cpp
template<int D>
struct RichPoint {
  Pos pos;
  float n[3];
  SHs<D> shs;
  float opacity;
  Scale scale;
  Rot rot;
};
```

출력 SoA:

- `pos[k]`
- `rot[k]`
- `scales[k]`
- `opacities[k]`
- `shs[k]`

이렇게 바꾸는 이유:

1. CUDA rasterizer가 `float* means3D`, `float* shs`, `float* opacities`처럼 SoA 포인터를 기대한다.
2. Morton order 정렬을 적용하기 쉽다.
3. 동일 tile에 들어갈 확률이 큰 Gaussian끼리 메모리 locality를 높일 수 있다.

즉, 이 단계는 단순 파싱이 아니라 viewer/rasterizer 친화적 재배열 단계다.

### Part 38. Morton order 정렬이 하는 일

이 부분은 단순 최적화처럼 보이지만 실제로는 메모리 접근 패턴을 바꾼다.

입력:

- 각 Gaussian의 3D 중심 `points[i].pos`

중간 처리:

1. scene bbox `minn/maxx` 계산
2. 각 점을 bbox 기준 `[0, 1]` 상대 좌표로 정규화
3. 21bit 정수 grid로 확장
4. x/y/z 비트를 interleave해서 Morton code 생성
5. Morton code 기준 정렬

출력:

- 공간적으로 가까운 Gaussian들이 SoA에서 비슷한 인덱스 근처로 모임

출력 사용처:

- 이후 CUDA rasterizer의 tile 기반 access locality 개선

중요한 구현상 주의점:

- bbox 축 길이가 0이면 `(maxx - minn)` 분모가 0이 되어 NaN/Inf가 생길 수 있다.
- 현재 코드는 이 경우를 방어하지 않는다.

### Part 39. 학습 저장 포맷 -> 렌더링 포맷으로 바뀌는 값들

`loadPly<D>()`는 저장된 raw parameter를 그대로 쓰지 않는다.

#### rotation

입력:

- 저장된 quaternion 4개

가공:

- 길이를 다시 계산해 정규화

출력:

- 단위 quaternion

출력 사용처:

- CUDA에서 3D covariance 계산
- GL ellipsoid shader

#### scale

입력:

- 저장된 `log(scale)` 값

가공:

- `exp()`

출력:

- 실제 positive scale

출력 사용처:

- CUDA `computeCov3D`
- GL ellipsoid shader

#### opacity

입력:

- 저장된 logit opacity

가공:

- `sigmoid()`

출력:

- `[0, 1]` 범위 opacity

출력 사용처:

- CUDA alpha compositing
- GL ellipsoid mode alpha thresholding

#### SH coefficients

입력:

- 학습 저장 순서의 SH coefficient

가공:

- rasterizer가 기대하는 RGB interleaved layout으로 재배치

출력:

- `shs[k].shs[j * 3 + c]`

출력 사용처:

- CUDA `computeColorFromSH`

### Part 40. 적재 이후 유지되는 영속 메모리와 프레임별 임시 메모리

#### 영속 메모리

- `pos_cuda`
- `rot_cuda`
- `scale_cuda`
- `opacity_cuda`
- `shs_cuda`
- `view_cuda`
- `proj_cuda`
- `cam_pos_cuda`
- `background_cuda`
- `rect_cuda`
- `imageBuffer` / `imageBufferCuda`

이들은 viewer lifetime 동안 유지된다.

#### 프레임별 내용만 바뀌는 메모리

- `view_cuda`
- `proj_cuda`
- `cam_pos_cuda`
- `imageBuffer` 내부 내용

#### lazy grow scratch arena

- `geomPtr`
- `binningPtr`
- `imgPtr`

이 3개는 `resizeFunctional()`을 통해 필요할 때만 커지고, 한 번 커지면 재사용된다.

즉, `CudaRasterizer::forward()`는 매 프레임 `cudaMalloc`을 새로 하지 않도록 함수형 allocator 콜백을 받는다.

### Part 41. PLY 로드 경로에 대한 리뷰 포인트

#### High. 헤더 파싱이 지나치게 취약하다

문제:

- header property 목록을 실제로 읽지 않는다.
- `count`를 사실상 3번째 줄에서만 추출한다.

영향:

- header 줄 순서가 조금만 달라도 파싱 실패 또는 잘못된 count 가능

#### Medium. body read 성공 여부를 검사하지 않는다

문제:

- `infile.read(...)` 뒤에 `gcount()`나 stream state 확인이 없다.

영향:

- 손상된 PLY를 읽을 때 부분 읽기 후도 그대로 진행할 수 있다.

#### Medium. peak memory footprint가 크다

문제:

- AoS CPU + SoA CPU + SoA GPU + GL duplicate가 한동안 공존한다.

영향:

- 큰 scene에서 startup memory spike 발생

#### Low. `savePly()`는 항상 degree-3 layout으로 저장한다

문제:

- `j = 1..15`, `f_rest_0..44`를 항상 쓴다.

영향:

- 낮은 SH degree로 연 경우에도 저장 포맷은 degree-3 container가 된다.

---

## 11. 추가 심화 리뷰: `MultiViewManager::onUpdate()` / `onRender()`

### Part 42. `MultiViewManager`는 단순 window manager가 아니라 “frame scheduler”다

이 클래스의 핵심은 subview를 단순히 보관하는 것이 아니다. 각 프레임마다 다음 4가지를 순서대로 관리한다.

1. 입력을 subview별로 자른다.
2. camera handler를 먼저 갱신한다.
3. 각 view를 offscreen RT에 렌더링한다.
4. GUI window 안에 RT를 배치하고 focus 상태를 갱신한다.

### Part 43. `onUpdate()` 상세 호출 순서

실제 순서는 아래와 같다.

```text
MultiViewBase::onUpdate(input)
  -> pause 토글 검사
  -> deltaTime 계산
  -> basic subviews update
  -> IBR subviews update
  -> sub-multiview update
```

IBR subview 하나 기준으로 보면:

```text
if view.active():
  subInput = view.isFocused() ? Input::subInput(...) : Input()
  if handler:
    handler->update(subInput, dt, viewport)
  cam = updateFunc(view, subInput, viewport, dt)
  if defaultUpdateFunc && handler:
    cam = handler->getCamera()
```

여기서 중요한 점은 두 가지다.

1. **handler update가 view update보다 먼저 실행된다.**
2. **기본 IBR update 함수를 쓰면 최종 eye camera는 handler가 결정한다.**

`gaussianViewer`는 정확히 이 기본 경로를 탄다.

- `addIBRSubView("Point view", gaussianView, ...)`
- `gaussianView->onUpdate()`는 사실상 no-op
- `addCameraForView("Point view", generalCamera)`

즉, Gaussian splat view의 카메라는 `GaussianView`가 아니라 `InteractiveCameraHandler`가 전적으로 결정한다.

### Part 44. `Input::subInput()`가 실제로 하는 일

입력:

- global input
- subview viewport

출력:

- viewport-local 좌표계로 변환된 input

가공:

- 마우스 좌표를 viewport origin 기준으로 평행이동
- viewport 밖이면 mouse button/scroll을 비움
- 옵션에 따라 keyboard도 비울 수 있음

중요한 점:

- `MultiViewBase::onUpdate()`는 view가 focused가 아니면 아예 빈 `Input()`를 넘긴다.
- 즉, subview는 “현재 포커스된 상태에서만” 입력을 받는다.

### Part 45. focus 상태는 update 단계가 아니라 render 단계에서 결정된다

`renderSubView()` 마지막 부분에서:

```cpp
subview.view->setFocus(showImGuiWindow(...))
```

가 호출된다.

즉, frame N의 focus 판단은 frame N의 render에서 갱신되고, frame N+1의 update에서 사용된다.

이 구조의 의미:

- 입력 라우팅은 1프레임 지연된 focus state를 쓴다.
- 의도적으로 GUI interaction과 input routing을 렌더 단계에서 합쳐 둔 구조다.

### Part 46. `onRender()` 상세 호출 순서

`MultiViewManager::onRender(win)`은 아래 순서로 동작한다.

1. window viewport bind
2. window clear
3. 상단 menu/gui draw
4. IBR subview들 render
5. basic subview들 render
6. nested multiview render
7. FPS counter update

즉, 실제 render 우선순위는 IBR subview가 먼저다.

### Part 47. `renderSubView()`는 실제로 세 단계 렌더링을 한다

subview 하나 기준:

1. **offscreen render**
   - `renderViewport = (0,0, rt.w, rt.h)`
   - `subview.render(...)`

2. **post render**
   - screenshot/video save
   - additional rendering callback
   - camera handler overlay render

3. **GUI presentation**
   - `showImGuiWindow(...)`
   - focus 갱신

즉, 화면에 보이는 것은 view가 window에 직접 그린 것이 아니라, 먼저 subview 전용 RT에 그린 결과를 ImGui window에 표시한 것이다.

### Part 48. IBR subview의 실제 render 경로는 한 번 더 간접화되어 있다

`IBRSubView::render()`는 직접 `view.onRenderIBR(...)`를 호출하지 않는다.

```text
IBRSubView::render()
  -> renderingMode->render(view, cam, renderViewport, rt.get())
```

기본 mode인 `MonoRdrMode::render()`는 다시:

1. 자신의 내부 `_destRT`에 `view.onRenderIBR(*_destRT, eye)` 수행
2. `_destRT.texture()`를 screen quad로 `optDest`에 복사

즉, IBR view는 실질적으로 다음 구조다.

```text
ViewBase::onRenderIBR()
  -> MonoRdrMode internal RT
  -> subview RT
  -> ImGui window
```

이중 RT 구조를 쓰는 이유:

- stereo/anaglyph 같은 다른 rendering mode와 인터페이스를 통일하기 위해서다.

### Part 49. `gaussianViewer`에서 실제로 어떤 객체가 무엇을 담당하는가

`Point view` 기준 역할 분담은 아래와 같다.

- `GaussianView`: novel view 이미지 생성
- `InteractiveCameraHandler`: eye camera 생성
- `MonoRdrMode`: IBR view 결과를 RT로 합성
- `MultiViewManager`: update/render scheduling, GUI focus, subview placement

`Top view`는 다르다.

- `SceneDebugView`가 basic subview로 등록된다.
- 따라서 `onRender(viewport)` 경로를 타고, IBR camera path를 타지 않는다.

### Part 50. `MultiViewManager` 경로에 대한 리뷰 포인트

#### High. Gaussian view의 camera source가 view 내부에 없다는 점을 놓치기 쉽다

문제:

- `GaussianView::onUpdate()`가 비어 있어서 얼핏 보면 camera 갱신이 없는 것처럼 보인다.

실제:

- camera는 전적으로 `InteractiveCameraHandler`에서 생성되고
- `MultiViewBase::onUpdate()`가 `fView.cam`에 주입한다.

영향:

- camera bug를 찾을 때 `GaussianView`만 보면 원인을 놓치기 쉽다.

#### Medium. focus는 render 단계에서 갱신되어 update 단계에서 소비된다

영향:

- 입력 라우팅을 디버깅할 때 한 프레임 temporal coupling을 이해해야 한다.

#### Medium. `MonoRdrMode`의 internal RT resize check가 주석 처리되어 있다

파일:

- `SIBR_viewers/src/core/view/RenderingMode.cpp`

문제:

- `_destRT->w() != w || _destRT->h() != h` 조건이 비활성화되어 있다.

영향:

- render resolution 변경이 생기면 stale size RT를 계속 쓸 가능성

#### Low. `renderSubView()`는 readback 기반 screenshot/video 경로 때문에 비싸다

영향:

- recording 기능이 켜진 경우 GPU->CPU stall이 프레임 타임에 직접 영향을 준다.

---

## 12. 추가 심화 리뷰: `CudaRasterizer::Rasterizer::forward`

### Part 51. `forward()`는 단일 커널이 아니라 4단계 파이프라인이다

겉보기에는 함수 하나지만 내부는 아래 4단계다.

```text
Rasterizer::forward
  -> scratch state bind/allocation
  -> FORWARD::preprocess(...)
  -> duplicateWithKeys + radix sort + identifyTileRanges
  -> FORWARD::render(...)
```

이 구조를 이해해야 `forward`의 입력과 출력이 왜 저렇게 생겼는지 보인다.

### Part 52. `forward()` 입력이 의미하는 것

입력 포인터들은 크게 5부류다.

#### 1) Gaussian geometry

- `means3D`
- `scales`
- `rotations`
- `opacities`

#### 2) Gaussian appearance

- `shs`
- `colors_precomp`

둘 중 하나만 있으면 된다.

#### 3) camera

- `viewmatrix`
- `projmatrix`
- `cam_pos`
- `tan_fovx`
- `tan_fovy`
- `width`
- `height`

#### 4) output / aux

- `out_color`
- `depth`
- `radii`

#### 5) memory providers

- `geometryBuffer`
- `binningBuffer`
- `imageBuffer`

이 마지막 3개가 중요하다. `forward`는 자기 내부에서 `cudaMalloc`을 직접 남발하지 않고, 외부에서 받은 arena grow callback을 통해 scratch 공간을 확보한다.

### Part 53. scratch state 구조체가 의미하는 것

#### `GeometryState`

역할:

- Gaussian 1개당 생기는 중간값 저장

필드 의미:

- `depths`: view space depth
- `clamped`: SH->RGB에서 음수 clamp 여부
- `internal_radii`: 화면 반경
- `means2D`: projected 2D center
- `cov3D`: 계산된 3D covariance
- `conic_opacity`: inverse 2D covariance + opacity packed
- `rgb`: preprocessed RGB
- `tiles_touched`: Gaussian이 덮는 tile 수
- `point_offsets`: duplicated instance 시작 offset prefix sum

#### `BinningState`

역할:

- tile-depth 정렬용 duplicated instance 목록

필드 의미:

- `point_list_keys_unsorted`
- `point_list_keys`
- `point_list_unsorted`
- `point_list`

즉, “Gaussian index를 tile별로 복제해서 depth 기준 정렬한 목록”을 담는다.

#### `ImageState`

역할:

- tile/pixel 단위 렌더링 보조 버퍼

필드 의미:

- `ranges`: tile별 sorted list 구간
- `n_contrib`: 픽셀에 실제 기여한 마지막 Gaussian index
- `accum_alpha`: 최종 transmittance 추적용

주의:

- 현재 구현은 `ImageState::fromChunk(width * height)`로 호출되어 `ranges`도 pixel 수만큼 할당한다.
- 논리적으로는 tile 수만큼이면 충분하므로 다소 과할당이다.

### Part 54. 1단계: `FORWARD::preprocess`

이 단계는 Gaussian 1개당 한 스레드에 가깝다.

입력:

- world-space Gaussian parameter
- camera
- viewport size

출력:

- `means2D`
- `depths`
- `cov3D`
- `rgb`
- `conic_opacity`
- `radii`
- `tiles_touched`

세부 처리:

1. **frustum test**
   - `in_frustum()`
   - 현재 구현은 사실상 `p_view.z > 0.2`만 강하게 본다.

2. **project**
   - `transformPoint4x4`
   - NDC -> pixel center

3. **3D covariance**
   - scale + rotation -> 3x3 covariance

4. **2D covariance**
   - view/proj에 맞춰 screen-space ellipse로 변환

5. **antialiasing 보정**
   - covariance에 추가 blur를 섞고 scale 조정

6. **conic inverse 계산**
   - 2D covariance inverse를 packed float3로 저장

7. **screen radius 계산**
   - eigenvalue 기반 3-sigma radius

8. **tile overlap 개수 계산**
   - `getRect(...)`
   - 몇 개 tile을 덮는지 계산

9. **색상 계산**
   - precomputed color가 없으면 SH -> RGB

즉, preprocess의 본질은 “3D Gaussian을 screen-space elliptical splat descriptor로 바꾸는 단계”다.

### Part 55. 2단계: tile-depth key 생성과 정렬

#### `duplicateWithKeys`

입력:

- Gaussian별 `means2D`, `depths`, `point_offsets`, `radii`

출력:

- duplicated `(tile, depth, gaussian_id)` 리스트

동작:

- Gaussian 하나가 여러 tile에 걸치면 tile 수만큼 복제한다.
- key는 `[ tile_id | depth_bits ]`
- value는 `gaussian_id`

중요한 점:

- float depth를 `uint32` 비트로 그대로 key 하위 32bit에 넣는다.
- 여기서 depth가 양수일 때 IEEE754 bit ordering을 이용해 정렬 순서를 얻는다.

#### `cub::DeviceRadixSort::SortPairs`

출력:

- tile 우선, depth 차선 순으로 정렬된 Gaussian instance 목록

출력 사용처:

- 다음 단계의 tile-local blending

### Part 56. 3단계: tile별 구간 분할

`identifyTileRanges()`는 정렬된 key 배열을 훑어:

- tile A는 `[startA, endA)`
- tile B는 `[startB, endB)`

형태의 range를 `imgState.ranges`에 기록한다.

이후 render kernel은 tile 하나당 자기 range만 보면 된다.

즉, global sort 결과를 tile-local work queue로 바꾸는 단계다.

### Part 57. 4단계: `FORWARD::render`

핵심 mapping:

- block 하나 = tile 하나
- thread 하나 = pixel 하나

현재 config:

- `BLOCK_X = 16`
- `BLOCK_Y = 16`
- 즉 tile size는 `16x16`, block size는 `256` threads

커널 내부 흐름:

1. 현재 block이 담당하는 tile과 pixel 범위 계산
2. tile의 Gaussian instance range 확인
3. 256개 단위로 Gaussian instance를 shared memory로 batch fetch
4. 각 pixel thread가 batch 안 Gaussian들을 차례대로 평가
5. conic matrix 기반으로 Gaussian weight 계산
6. alpha compositing 누적
7. transmittance `T`가 충분히 작아지면 early exit
8. 마지막에 RGB와 depth를 write back

### Part 58. 픽셀 누적 공식의 실제 의미

각 pixel thread는 아래 상태를 가진다.

- `T`: 현재까지 남은 transmittance
- `C[3]`: 누적 RGB
- `expected_invdepth`
- `last_contributor`

각 Gaussian에 대해:

1. 픽셀에서 Gaussian 중심까지의 2D 차이 `d`
2. inverse covariance `conic`로 exponent `power` 계산
3. `alpha = opacity * exp(power)` 계산 후 clamp
4. `C += feature * alpha * T`
5. `expected_invdepth += (1 / depth) * alpha * T`
6. `T *= (1 - alpha)`

즉, 논문식 front-to-back alpha compositing을 tile-sorted order로 수행한다.

### Part 59. `out_color`의 메모리 레이아웃

최종 write:

```cpp
out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
```

즉, output은 HWC가 아니라 **CHW planar float layout**이다.

이 점이 `BufferCopyRenderer`의 shader와 정확히 맞물린다.

- R plane 전체
- G plane 전체
- B plane 전체

### Part 60. `depth` 인자의 실제 의미

이 API 이름은 `depth`지만 실제 write 값은:

```cpp
expected_invdepth += (1 / depths[id]) * alpha * T;
invdepth[pix_id] = expected_invdepth;
```

즉, **raw depth가 아니라 composited inverse depth**다.

이건 Python 경로와 의미가 맞다.

- `train.py`는 `render_pkg["depth"]`를 `invDepth`로 사용한다.

로컬 Gaussian viewer는 이 포인터에 `nullptr`를 넘겨 depth output을 쓰지 않는다.

### Part 61. `forward()`와 viewer side scratch arena의 연결

`GaussianView`는 아래 콜백을 만든다.

- `geomBufferFunc`
- `binningBufferFunc`
- `imgBufferFunc`

각 콜백은 요청 크기 `N`이 현재 capacity보다 크면:

- 기존 포인터 free
- `cudaMalloc(2 * N)`
- capacity 갱신

을 수행한다.

즉, 큰 scene에서 한 번 커진 scratch buffer는 이후 프레임에 재사용된다. 이 설계 덕분에 프레임마다 불필요한 allocator 비용이 줄어든다.

### Part 62. `forward()` 경로에 대한 리뷰 포인트

#### High. 현재 viewer 호출부와 submodule 헤더 시그니처가 일치하지 않는다

파일:

- `SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.cpp`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h`

관찰:

- viewer 호출부는 `nullptr, rects, boxmin, boxmax` 같은 추가 인자를 넘긴다.
- 하지만 현재 읽힌 submodule 헤더의 `forward(...)` 선언에는 그 인자가 없다.

의미:

- 로컬 build가 확장된 rasterizer API를 기대하고 있을 가능성
- 또는 현재 코드베이스 상태가 build-consistent하지 않을 가능성

이건 문서화만의 문제가 아니라 실제 유지보수 리스크다.

#### High. near culling이 입력 znear를 쓰지 않고 `0.2f` 상수에 묶여 있다

파일:

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h`

문제:

- `in_frustum()`가 `p_view.z <= 0.2f`면 무조건 false

영향:

- 카메라의 실제 `znear`와 무관하게 0.2보다 가까운 Gaussian은 제거된다.

#### Medium. x/y frustum culling은 사실상 비활성화되어 있다

문제:

- `p_proj.x/y` 체크 코드가 주석 처리되어 있다.

영향:

- offscreen Gaussian도 preprocess를 거친 뒤 tile rect가 0이 될 때까지 비용을 쓴다.

#### Medium. duplicated instance 수가 많으면 메모리 사용량이 급증한다

문제:

- Gaussian 하나가 큰 radius를 가지면 여러 tile로 복제된다.

영향:

- `num_rendered`가 원래 Gaussian 수보다 훨씬 커질 수 있다.
- `BinningState`와 radix sort buffer가 크게 증가한다.

#### Medium. `ImageState::ranges`는 과할당된다

문제:

- tile 수가 아니라 pixel 수 기준으로 잡힌다.

영향:

- 큰 해상도에서 불필요한 scratch memory 낭비

#### Low. `alpha < 1/255` cutoff와 `T < 1e-4` early exit는 품질/속도 tradeoff다

의미:

- 아주 약한 기여는 버리고
- 충분히 opaque해진 픽셀은 더 이상 계산하지 않는다.

장점:

- 성능 향상

단점:

- 극단적인 투명 구조에서 미세한 tail contribution은 잘린다.
