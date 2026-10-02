# IGRIS-C v2 motion canonical form (GMR PKL)

파일은 Python `pickle`로 직렬화한 `dict`이며, NumPy 배열을 포함한다. 로봇 모델은 [`assets/robros_igris_c_v2/igris_c_v2.xml`](assets/robros_igris_c_v2/igris_c_v2.xml), 로봇 식별자는 `robros_igris_c_v2`이다.

## 공통 규칙

- `T` = 프레임 수, `K` = `len(link_body_list)`. 전체 키바디를 포함하면 `K=33`이다. `fps`는 모션마다 다른 양의 값이며 시간 간격은 `dt=1/fps`이다. 별도 timestamp 배열은 없다. MuJoCo XML의 `timestep=0.005`는 저장 모션의 fps가 아니다.
- canonical 수치 배열은 NumPy `float32` (`fp32`), `fps`는 Python `float`로 저장한다. 기존 GMR 파일은 `float64`일 수도 있으므로 읽을 때 dtype을 확인하고 변환 결과를 `float32`로 맞춘다.
- MuJoCo 좌표계: 월드 `+Z`가 위, 위치 단위는 m, 회전 관절은 rad, 선속도는 m/s, 각속도는 rad/s. 쿼터니언 배열의 마지막 차원 순서는 **`[w, x, y, z]`**이다. SciPy `Rotation`에 전달할 때는 `[x, y, z, w]`로 재배열한다.
- 자유 베이스는 `base_link`이고, `qpos[t] = concat(root_pos[t], root_rot[t], dof_pos[t])`로 재구성한다. shape은 `(T, 38)`이며 `qpos[:3]`는 월드 위치, `qpos[3:7]`은 월드 기준 베이스 방향, `qpos[7:]`은 아래 31개 관절각이다. `root_rot`은 첫 번째 키바디인 `Link_Waist_Yaw`의 회전과 같지 않을 수 있다.

## 최상위 key-value

| key | 타입 / shape | 의미 |
| --- | --- | --- |
| `fps` | `float` | 프레임/초 (`> 0`). |
| `root_pos` | `float32 (T, 3)` | `base_link` 원점의 월드 XYZ 위치. |
| `root_rot` | `float32 (T, 4)` | `base_link`의 월드 방향, `wxyz`. |
| `root_vel` | `float32 (T, 3)` | 월드 XYZ 선속도. |
| `root_angvel` | `float32 (T, 3)` | 베이스 상대 회전에서 계산한 XYZ 각속도. 아래 계산식 참조. |
| `dof_pos` | `float32 (T, 31)` | MuJoCo `qpos[7:]`의 관절각, 아래 순서. |
| `dof_vel` | `float32 (T, 31)` | 같은 순서의 관절 각속도. |
| `keybody_pos_world` | `float32 (T, K, 3)` | `link_body_list` 순서의 MuJoCo body 원점 월드 위치 (`data.xpos`). |
| `keybody_rot_world` | `float32 (T, K, 4)` | 같은 body의 월드 방향 (`data.xquat`), `wxyz`. |
| `keybody_pos_local` | `float32 (T, K, 3)` | 같은 위치를 `base_link` 좌표계로 변환. |
| `keybody_rot_local` | `float32 (T, K, 4)` | 같은 방향을 `base_link` 좌표계로 변환, `wxyz`. **각 링크의 부모 관절 기준 회전이 아니다.** |
| `keybody_pos` | `float32 (T, K, 3)` | `keybody_pos_world`와 값이 같은 레거시 별칭. |
| `link_body_list` | `list[str]`, 길이 `K` | `keybody_*`의 두 번째 축에 대응하는 body 이름. |
| `local_body_link_body_list` | `None` | 이 형식에서는 사용하지 않음. |
| `local_body_pos` | `None` | 이 형식에서는 사용하지 않음. |
| `retarget_meta` | `dict` (선택) | 생성 경로별 추가 메타데이터. 고정 키는 없으며 포즈 재구성에 필요하지 않음. |

키바디를 생략한 기존 파일은 `K=0`, `link_body_list=[]`, `keybody_*`의 두 번째 차원이 0일 수 있다. 전체 키바디가 필요한 변환 결과에는 아래 33개 이름을 순서대로 사용한다.

## 관절 및 키바디 순서

`dof_pos`/`dof_vel` 열 인덱스 (`0`부터 시작):

```text
 0 Joint_Waist_Yaw           1 Joint_Waist_Roll          2 Joint_Waist_Pitch
 3 Joint_Hip_Pitch_Left      4 Joint_Hip_Roll_Left       5 Joint_Hip_Yaw_Left
 6 Joint_Knee_Pitch_Left     7 Joint_Ankle_Pitch_Left    8 Joint_Ankle_Roll_Left
 9 Joint_Hip_Pitch_Right    10 Joint_Hip_Roll_Right     11 Joint_Hip_Yaw_Right
12 Joint_Knee_Pitch_Right   13 Joint_Ankle_Pitch_Right  14 Joint_Ankle_Roll_Right
15 Joint_Shoulder_Pitch_Left 16 Joint_Shoulder_Roll_Left 17 Joint_Shoulder_Yaw_Left
18 Joint_Elbow_Pitch_Left   19 Joint_Wrist_Yaw_Left     20 Joint_Wrist_Roll_Left
21 Joint_Wrist_Pitch_Left   22 Joint_Shoulder_Pitch_Right 23 Joint_Shoulder_Roll_Right
24 Joint_Shoulder_Yaw_Right 25 Joint_Elbow_Pitch_Right  26 Joint_Wrist_Yaw_Right
27 Joint_Wrist_Roll_Right   28 Joint_Wrist_Pitch_Right  29 Joint_Neck_Yaw
30 Joint_Neck_Pitch
```

`link_body_list`의 인덱스 (`keybody_*[:, i]`):

```text
 0 Link_Waist_Yaw           1 Link_Waist_Roll          2 Link_Waist_Pitch
 3 Link_Hip_Pitch_Left      4 Link_Hip_Roll_Left       5 Link_Hip_Yaw_Left
 6 Link_Knee_Pitch_Left     7 Link_Ankle_Pitch_Left    8 Link_Ankle_Roll_Left
 9 Link_Hip_Pitch_Right    10 Link_Hip_Roll_Right     11 Link_Hip_Yaw_Right
12 Link_Knee_Pitch_Right   13 Link_Ankle_Pitch_Right  14 Link_Ankle_Roll_Right
15 Link_Shoulder_Pitch_Left 16 Link_Shoulder_Roll_Left 17 Link_Shoulder_Yaw_Left
18 Link_Elbow_Pitch_Left   19 Link_Wrist_Yaw_Left     20 Link_Wrist_Roll_Left
21 Link_Wrist_Pitch_Left   22 Left_Hand               23 Link_Shoulder_Pitch_Right
24 Link_Shoulder_Roll_Right 25 Link_Shoulder_Yaw_Right 26 Link_Elbow_Pitch_Right
27 Link_Wrist_Yaw_Right    28 Link_Wrist_Roll_Right   29 Link_Wrist_Pitch_Right
30 Right_Hand              31 Link_Neck_Yaw           32 Link_Neck_Pitch
```

손가락 관절은 `dof_pos`에 없다. `Left_Hand`/`Right_Hand`는 관절 열이 아니라 키바디 이름이다.

## 파생값 계산 및 변환

`R(q)`를 `wxyz` 쿼터니언이 나타내는 회전, `p=root_pos[t]`, `q=root_rot[t]`라고 하면 다음과 같다. `keybody_*_local`은 **베이스 좌표계** 값이다.

```text
keybody_pos_local[t, i] = R(q)^(-1) · (keybody_pos_world[t, i] - p)
keybody_rot_local[t, i] = q^(-1) ⊗ keybody_rot_world[t, i]
keybody_pos[t, i]       = keybody_pos_world[t, i]
```

`root_vel`과 `dof_vel`은 첫/마지막 프레임에서 1차 차분, 내부에서 중심 차분을 쓴다: `v[0]=(x[1]-x[0])/dt`, `v[t]=(x[t+1]-x[t-1])/(2dt)`, `v[T-1]=(x[T-1]-x[T-2])/dt`. `root_angvel[t]`은 내부에서 `log(R(q[t-1])^-1 R(q[t+1]))/(2dt)`, 양 끝에서 각각 `log(R(q[0])^-1 R(q[1]))/dt`와 `log(R(q[T-2])^-1 R(q[T-1]))/dt`이다 (`log`는 SciPy `Rotation.as_rotvec`). 상대 회전의 기준은 첫 번째 쿼터니언이므로 이 배열을 월드 축 각속도로 취급하지 않는다. `T=1`이면 속도는 0이다.

다른 형식에서 변환할 때는 관절 이름을 위 31열로 **이름 기반 재배열**한 뒤, 위치·방향·관절각으로 `qpos`를 만들고 `fps`에 맞춰 속도를 계산한다. 키바디가 필요하면 해당 XML에 `qpos`를 넣고 `mj_forward` 후 `data.xpos`/`data.xquat`를 위 33개 body 이름 순으로 추출한다. 공통 생성 로직: [`motion_utils.py`](general_motion_retargeting/utils/motion_utils.py).

## Object 및 terrain 표준 형식

학습 데이터에서 object와 terrain은 동일한 **scene entity track** 형식을 쓴다. 각 entity는 고정된 로컬 좌표계의 USD 형상과 시간에 따른 월드 pose를 가진다. Object는 이동·회전할 수 있고, terrain은 보통 고정되어 있지만 움직이는 플랫폼도 같은 형식으로 표현한다. 이 pose track은 강체 이동만 나타내므로 프레임마다 형상이 변하는 변형체는 별도 geometry animation 규격이 필요하다.

### 시간 및 pose 규칙

- Track은 모션과 같은 `T` 및 `fps`를 사용한다. 프레임 `t`의 시간은 `t / fps` 초이며, actor와 entity의 프레임은 같은 시각을 가리킨다. entity가 없는 프레임도 행을 생략하지 않는다.
- Pose는 6-DoF 강체 pose `(translation, rotation)`로 저장한다. 위치는 월드 XYZ 미터, 회전은 월드 방향 `wxyz` 쿼터니언이다. 따라서 두 배열은 각각 `(T, 3)`과 `(T, 4)`이고, 의미상 하나의 6-DoF pose를 이룬다. 회전을 3개 오일러 각으로 저장하지 않는다.
- 정적 terrain도 canonical 출력에서는 모든 프레임에 pose를 기록한다. 고정 terrain은 모든 프레임의 `pose_pos`/`pose_rot` 값이 같아야 한다. 원본 저장 공간을 줄이는 것은 허용하지만 학습용 canonical 변환 결과는 `(T, ...)`로 펼친다.
- Pose는 USD asset의 지정된 entity root prim 좌표계를 월드 좌표계에 놓는 변환이다. asset 내부 vertex와 자식 prim의 좌표는 USD에 보존한다. USD stage의 `metersPerUnit` 및 `upAxis`를 읽어 이 문서의 미터 및 `+Z` up 좌표계로 변환한 뒤 저장한다.
- 쿼터니언은 이 문서 전체와 동일하게 `[w, x, y, z]` 순서다. 프레임 간 부호가 임의로 뒤집히지 않게 연속성을 맞춘다 (`dot(q[t], q[t-1]) >= 0`).

### 파일 분리 및 디렉터리 구조

Actor motion PKL과 object/terrain 데이터는 별도 파일 및 별도 디렉터리에 저장한다. 모션 PKL에는 `scene_entities`를 넣지 않는다. object와 terrain은 공통 record 형식을 사용하며, 한 clip의 모든 entity record를 scene track PKL 하나에 저장한다. mesh/scene geometry는 별도의 USD asset 디렉터리에 둔다.

```text
dataset_root/
  motion/
    <clip_id>.pkl                 # actor motion only
  scene_tracks/
    <clip_id>.pkl                 # that clip's object and terrain tracks
  robot/
    igris_c.mjb                   # self-contained MuJoCo robot model for playback
  usd_assets/
    objects/
      <asset_id>.usd, .usda, .usdc, or .usdz
    terrain/
      <asset_id>.usd, .usda, .usdc, or .usdz
```

`motion/<clip_id>.pkl`과 `scene_tracks/<clip_id>.pkl`의 `<clip_id>`는 같아야 한다. scene track PKL의 `usd_asset`은 `dataset_root` 기준 상대 경로로 기록하며, USD 파일은 `usd_assets/objects/` 또는 `usd_assets/terrain/`에 저장한다. Viser 재생용 `robot/igris_c.mjb`는 로봇 mesh를 포함하므로 canonical 디렉터리 외부의 XML/mesh를 참조하지 않는다. 새 변환 결과의 `retarget_meta.robot_model`에는 이 상대 경로를 기록한다. scene track 파일이 없으면 해당 clip에 object/terrain track이 없는 것으로 해석한다. scene track PKL의 최상위 키와 값은 다음과 같다.

| key | 타입 | 의미 |
| --- | --- | --- |
| `fps` | `float` | 연결된 motion PKL과 같은 프레임/초. |
| `scene_entities` | `list[dict]` | object 및 terrain entity record 목록. 없으면 빈 목록으로 간주한다. |

Record 리스트 순서는 clip 내내 고정한다. Record 형식은 object와 terrain에 공통이다.

| key | 타입 / shape | 의미 |
| --- | --- | --- |
| `entity_id` | `str` | clip 안에서 유일하고 시간에 따라 변하지 않는 ID. |
| `entity_type` | `str` | 표준값 `object` 또는 `terrain`. |
| `usd_asset` | `str` | mesh/scene 형상을 가진 USD 파일의 경로. dataset root 기준 상대 경로로 기록한다. 확장자는 `.usd`, `.usda`, `.usdc` 또는 `.usdz`일 수 있다. |
| `usd_prim_path` | `str` | asset에서 pose를 적용할 entity root prim의 절대 prim path (예: `/Object`). |
| `pose_pos` | `float32 (T, 3)` | `usd_prim_path` 원점의 월드 위치 (m). |
| `pose_rot` | `float32 (T, 4)` | 해당 root prim의 월드 방향, `wxyz`. |
| `mobility` | `str` | 표준값 `static` 또는 `dynamic`. `static` entity의 pose는 모든 프레임에서 일정해야 한다. |
| `semantic_label` | `str` (선택) | 학습에 사용할 의미 클래스명 (예: `box`, `floor`, `ramp`). |

동일 clip 내 여러 entity는 각자 record를 하나씩 가진다. 각 record의 `T`는 actor motion의 `T`와 같아야 하며, 순서 정렬이나 batch tensor가 필요하면 `entity_id`를 보존한 채 별도 전처리 단계에서 만든다. Record 리스트 형식은 clip마다 entity 수가 달라도 schema를 유지할 수 있게 한다.

### USD mesh 및 transform 규칙

- USD는 형상과 형상에 속하는 정적 정보를 담는다. Mesh topology, vertex, material, 로컬 transform 및 필요하면 collision용 geometry를 USD asset에 둔다. 프레임별 이동/회전은 pickle의 pose track에 둔다.
- `usd_prim_path`는 asset 내부의 안정된 root prim이어야 한다. pose를 적용할 때 해당 prim의 USD 내부 transform을 보존하면서 entity pose를 바깥 월드 transform으로 적용한다. 프레임별 pose를 USD animation과 pickle 양쪽에 중복 기록하지 않는다.
- 여러 mesh/prim으로 구성된 하나의 물체나 terrain은 하나의 USD root prim 아래에 묶고 하나의 pose track으로 표현한다. 별도로 움직이는 구성 요소는 각각 entity record로 나눈다.
- asset 경로는 dataset 내에서 재현 가능하게 해석되어야 한다. 파일을 찾을 수 없는 절대 경로, DCC 프로그램 전용 참조, 외부 네트워크 URL은 canonical 데이터에 사용하지 않는다. USD의 외부 reference/texture도 asset과 함께 배포하고 상대 경로로 유지한다.

예시 (`scene_tracks/<clip_id>.pkl`의 내용; actor motion PKL과 별도 파일):

```python
{
    "fps": 30.0,
    "scene_entities": [
        {
            "entity_id": "box_001",
            "entity_type": "object",
            "usd_asset": "usd_assets/objects/box.usda",
            "usd_prim_path": "/Object",
            "pose_pos": float32_array(shape=(120, 3)),
            "pose_rot": float32_array(shape=(120, 4)),  # wxyz
            "mobility": "dynamic",
            "semantic_label": "box",
        },
        {
            "entity_id": "floor",
            "entity_type": "terrain",
            "usd_asset": "usd_assets/terrain/room.usd",
            "usd_prim_path": "/Terrain",
            "pose_pos": float32_array(shape=(120, 3)),
            "pose_rot": float32_array(shape=(120, 4)),  # same pose on every frame
            "mobility": "static",
            "semantic_label": "floor",
        },
    ],
}
```

`float32_array(...)`는 shape을 설명하기 위한 표기이며 실제 값은 NumPy `float32` 배열이다. Object와 terrain 모두 위 record로 변환하며, USD는 geometry source of truth, `pose_pos`/`pose_rot`은 시간에 따른 pose source of truth로 삼는다. Scene track PKL의 `fps`와 각 pose 배열의 `T`는 같은 clip의 motion PKL과 일치해야 한다.

### Viser 재생

`dataset_root` 안에 `motion/`, `scene_tracks/`, `usd_assets/`, `robot/igris_c.mjb`가 있으면 원본 UMR NPZ나 OMOMO source 디렉터리 없이 canonical clip을 재생할 수 있다. `scene_tracks/<clip_id>.pkl`이 없는 clip은 로봇만 재생한다.

```bash
bash scripts/viewer_canonical.sh
bash scripts/viewer.sh
```
