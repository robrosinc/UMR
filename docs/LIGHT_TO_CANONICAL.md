# Light form → canonical

## NAS 배포와 사용자 설치

NAS에는 각 데이터셋의 **light form 루트 전체**(`format.json`, `clips/`,
`scenes/`, `assets/`)를 올린다. NPZ만 복사하면 로봇 XML, 메시, 오브젝트 및
scene 참조가 끊긴다. 사용자는 원하는 위치로 디렉터리 전체를 복사한 뒤
canonical로 확장한다. 클립 안의 파일 참조는 NPZ 기준 상대 경로다.

UMR 저장소 전체나 리타게팅 환경은 필요하지 않다. 배포할 실행 파일은
`scripts/expand_light_to_canonical.py`와 `requirements-light-expand.txt` 두 개면
된다. 이 스크립트는 UMR 내부 모듈, 원본 데이터, `sample_data/`, 저장소의
`assets/`를 import하거나 참조하지 않는다. 별도의 Python 3.12 환경에서
다음처럼 실행한다. 의존성 버전은 Python 3.12, Linux x86_64에서 동작을
확인한 조합이다.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-light-expand.txt
python expand_light_to_canonical.py \
  --input-dir /data/igris_c_hiphi \
  --output-dir /data/canonical_hiphi
```

UMR 저장소 안에서 실행한다면 마지막 명령의 스크립트 경로를
`scripts/expand_light_to_canonical.py`로 바꾼다. 셸 래퍼도 사용할 수 있다.
최신 `requirements-umr.txt`로 설치한 UMR 환경이라면 expand 의존성도
포함되므로 별도 설치가 필요 없다. 이전에 만든 환경에는 변경된
`requirements-umr.txt`를 다시 설치하거나 `usd-core==26.8`을 추가한다.
활성화된 가상환경의 `python3`를 기본으로 사용하며,
`LIGHT_EXPAND_PYTHON`, `LIGHT_INPUT_DIR`, `CANONICAL_OUTPUT_DIR` 환경 변수로
실행 환경과 경로를 지정할 수 있다. Python 진입점은 `--limit N`(정렬된 처음
N개), `--clip CLIP_ID`(NPZ 파일명에서 `.npz`를 뺀 이름),
`--clip-path RELATIVE_NPZ`(`clips/` 기준 상대 경로로 정확히 한 클립)를 지원한다.
선택한 클립 집합을 바꿔 재실행할 때는 별도 출력 디렉터리를 사용한다.

```bash
LIGHT_EXPAND_PYTHON=/path/to/.venv/bin/python \
LIGHT_INPUT_DIR=/data/igris_c_hiphi \
CANONICAL_OUTPUT_DIR=/data/canonical_hiphi \
bash scripts/expand_light_to_canonical.sh
```

The converter has no source-dataset switch. It reads only portable light-form
fields, writes `motion/*.pkl`, `scene_tracks/*.pkl`, `robot/*.mjb`,
`usd_assets/`, and `objects_manifest.json`, and keeps every reference inside
the canonical output directory. Use a separate output directory per dataset.

## Current light-form v1 contract

- `clips/*.npz` contains `light_form_version=1`, `qpos`, `fps`,
  `robot_joint_names`, and a clip-relative `robot_xml`.
- If moving objects are present, all five fields are required:
  `light_object_names`, `light_object_xml_refs`, `light_object_scales`,
  `light_object_pos`, `light_object_rot`. Tracks are already in the retarget
  coordinate frame and sampled at the clip's frame rate.
- Object XML templates have visual mesh geoms in group 2 and collision mesh
  geoms in group 3. Their mesh files are package-local. The NPZ scale is the
  **final** uniform scale for the entire object; the converter never consults
  source-specific `smpl_scale`, source paths, or original trajectory files.
- `light_scene_ref`, when present, points to a version 1 scene JSON with
  `static_entities`. This implementation supports `terrain` `plane` entries.

An object base USD contains shared visual and collision geometry. Each distinct
object scale gets a small instance USD with a relative reference to that base
and one scale transform. `scene_tracks` points at the instance USD; it stores
the supplied world poses without scaling positions again. Both USD files must
be transferred together. New source adapters must normalize their geometry,
units, object tracks, and terrain to this light contract before expansion.

Other scene primitives or geometry encodings require an explicit light schema
extension and corresponding expander support. The current implementation is
verified against OMOMO, HiPHI, and object-free AMASS, Bones Seed, and LAFAN1
light form.
