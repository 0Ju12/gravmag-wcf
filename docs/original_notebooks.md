# 원본 노트북과의 대응표

| 이 저장소 | 원본 노트북의 셀·함수 | 비고 |
|---|---|---|
| `io.read_asc_grid` | `read_custom_asc_grid` | 동일 동작, print 제거 |
| `io.read_xyz_grid` | `read_xyz_to_grid` | 동일 (북쪽 행부터 정렬) |
| `io.read_grid` | `read_file_auto` | |
| `io.crop_grid` | `crop_grid`, `read_xyz_crop_to_grid` | 자르기 좌표를 인자로 |
| `io.read_bln` | `read_bln` | 오일필드 경계선(흰 선) |
| `io.grid_to_xyz` | `grid_to_xyz_df`, `grid_to_dataframe` | 엑셀 저장 전 단계 |
| `wcf.normalize` | `normalize` | z-score × 10 |
| `wcf.remove_mean` | `remove_trend_mean` | |
| `wcf.center_crop_common` | `auto_align_center_multi` | |
| `wcf.correlation_spectrum` | `compute_cck_with_res` | CC(k) = cos(Δθ) |
| `wcf.slfi_dlfi` | `compute_slfi_dlfi_B` (Mode B) | SLFI = D+F (CC>0), DLFI = F−D (CC<0) |
| `wcf.retained_power` | `cck_filter_maps_and_stats_3` 내부 | |
| `threshold.energy_curves` | `energy_analysis` (Step 04) | |
| `threshold.knee_point` | 상향 연속 높이 선택 (논문 Fig. 4) | 원본은 그래프로 판단 → 함수화 |
| `validate.corr_safe` | `corr_safe` (Validation) | Pearson·Spearman |
| `validate.polygon_mask` | `build_bln_mask_on_grid` | |
| `validate.resample_to_grid` | `resample_pmd_slice_to_slfi_grid` | 3D 밀도 모델 단면 → 지표 격자 |
| `validate.depthwise_correlation` | PMD vs SLFI 깊이별 상관 루프 | |
| `validate.sample_profile` | `sample_line_xy`, `make_grid_interpolator` | 측선 추출 |
| `validate.best_shift` | (측선 위치 보정 — 원본 별도 작업) | 재구현 |
| `wcf.upward_continue` | (원본은 상향 연속된 격자를 입력으로 사용) | 합성 예제용 재구현 |

원본 노트북에는 위 계산 외에 ipywidgets 슬라이더 화면, 양수·음수 영역별 등고선 그림, 결과 엑셀 저장, 공동연구 기관 모델(측선·3D 밀도) 비교 그림이 들어 있습니다. 이 부분은 데이터에 묶여 있어 제외했습니다.
