//! calcDisparityByZNCC の Rust 実装（core/disparity.py から呼ばれる内部拡張）。
//!
//! core/_disparity_reference.py（凍結された Python 参照実装）と同じ結果になるように
//! 移植している。境界条件・タイ処理・浮動小数の評価順まで意図的に複製した箇所は、
//! 各関数のコメントに理由を記す（`checks/verify_disparity_rust.py` で照合する）。
//! 丸めの違いにより、実写では近接ピークの 1 画素ほどが食い違うことがある
//! （詳細は checks/verify_disparity_rust.py の docstring）。
//!
//! 元の Python 実装は TiltedOpticsDepthIntro（MIT, Copyright (c) 2025 aoi1115）のコードを元にしている。
//! 著作権表示と許諾文はリポジトリ直下の NOTICE を参照。

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// BORDER_REFLECT_101 で座標を折り返す（OpenCV既定のSobel境界処理と同一: gfedcb|abcdefgh|gfedcba）。
/// n<=1（1行/1列画像）は自己参照になり定義不能なため 0 に丸める（縮退画像はどのみち有効視差が
/// 出ない領域なので、境界処理の厳密性より panic 回避を優先する）。
fn reflect_101(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n = n as isize;
    let mut idx = i;
    while idx < 0 || idx >= n {
        if idx < 0 {
            idx = -idx;
        } else {
            idx = 2 * (n - 1) - idx;
        }
    }
    idx as usize
}

/// Sobel(3x3, dx=1, dy=0) の水平勾配を計算し convertScaleAbs 相当（|value| を 0..255 にクランプ）
/// した u8 勾配画像を返す。入力が uint8・カーネルが整数のため、Sobel出力は常に整数値であり、
/// abs+clampは丸め曖昧性なく厳密に整数演算で再現できる（cv2.convertScaleAbs の cvRound は
/// 整数入力に対しては no-op）。
fn sobel_dx_abs(src: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w];
    for y in 0..h {
        let ym1 = reflect_101(y as isize - 1, h);
        let yp1 = reflect_101(y as isize + 1, h);
        for x in 0..w {
            let xm1 = reflect_101(x as isize - 1, w);
            let xp1 = reflect_101(x as isize + 1, w);
            // Sobel dx カーネル: [[-1,0,1],[-2,0,2],[-1,0,1]]
            let v: i32 = -(src[ym1 * w + xm1] as i32) + (src[ym1 * w + xp1] as i32)
                - 2 * (src[y * w + xm1] as i32)
                + 2 * (src[y * w + xp1] as i32)
                - (src[yp1 * w + xm1] as i32)
                + (src[yp1 * w + xp1] as i32);
            out[y * w + x] = v.abs().min(255) as u8;
        }
    }
    out
}

/// cv2.matchTemplate(image=red_patch, templ=blue_patch, TM_CCOEFF_NORMED) の返り値を、
/// 整数和から再現する（OpenCV の f32 累積とは 1〜数 ulp の丸め差が出る）。
///
/// 実機検証済みの縮退分散挙動（uint8 入力なので分散ゼロは浮動小数イプシロンではなく厳密な
/// 整数一致で判定できる）:
///   - templ（blue）側の分散が厳密0 -> 常に 1.0（image側の値に関わらず）
///   - image（red）側のみ分散0     -> 常に 0.0
///
/// 通常ケースは N*Σrb - Σr*Σb を分子、sqrt((N*Σr²-Σr²)*(N*Σb²-Σb²)) を分母とする
/// （Σ(r-r̄)(b-b̄) の N倍に等しい代数的簡略形。平均を経由せず整数和のみで厳密計算できる）。
#[inline]
fn zncc(n: i64, sum_r: i64, sum_r2: i64, sum_b: i64, sum_b2: i64, sum_rb: i64) -> f32 {
    let denom_r_sq = n * sum_r2 - sum_r * sum_r;
    let denom_b_sq = n * sum_b2 - sum_b * sum_b;
    if denom_b_sq == 0 {
        return 1.0;
    }
    if denom_r_sq == 0 {
        return 0.0;
    }
    let numerator = (n * sum_rb - sum_r * sum_b) as f64;
    let denominator = ((denom_r_sq as f64) * (denom_b_sq as f64)).sqrt();
    (numerator / denominator) as f32
}

/// 1行分（y固定）の視差を計算し、出力バッファ（幅w、NaN初期化済み）の該当行へ書き込む。
/// rayon での行並列化はこの関数を行ごとに独立呼び出しすることで実現する（出力は disjoint な
/// 行スライスにのみ書き込むため、データレースが構造的に発生しない）。
/// 引数過多は許容: ホットパスのため構造体詰め替えより素通しを優先（内部関数・呼び出し元は1箇所）。
#[allow(clippy::too_many_arguments)]
fn compute_disparity_row(
    y: usize,
    out_row: &mut [f32],
    edge_red: &[u8],
    edge_blue: &[u8],
    w: usize,
    kernel_size: usize,
    search_range_x: usize,
    th_value: f64,
    k_size: usize,
    s_range: usize,
    x_start: isize,
    x_end: isize,
) {
    debug_assert_eq!(out_row.len(), w);
    if x_end <= x_start {
        return; // 有効な x 範囲なし（全 NaN のまま）
    }
    let n = (kernel_size * kernel_size) as i64;

    // 行ごとに1回だけ確保し、画素ごとに上書き再利用する（全 search_range_x 要素を
    // 毎回書き切るため、前画素の値が残っていても正しさに影響しない）。
    // エッジ画素数分（実写1フレームで数千〜1万超）ヒープ確保していたのを行数分に削減し、
    // 30fps バジェットでの p95 ジッタを抑える最適化（実測ベンチで効果を確認した最適化）。
    let mut sim_array = vec![0f32; search_range_x];

    for x in (x_start as usize)..(x_end as usize) {
        // THRESH_TOZERO(EDGE_TH) 相当: src > th_value のときのみ残す。0 なら弱エッジとしてスキップ。
        let center = edge_red[y * w + x];
        let after_threshold = if (center as f64) > th_value {
            center
        } else {
            0
        };
        if after_threshold == 0 {
            continue;
        }

        // 固定される red パッチの統計（shift に依らず1回だけ計算）
        let red_row0 = y - k_size;
        let red_col0 = x - k_size;
        let mut sum_r: i64 = 0;
        let mut sum_r2: i64 = 0;
        for ky in 0..kernel_size {
            let base = (red_row0 + ky) * w + red_col0;
            for kx in 0..kernel_size {
                let rv = edge_red[base + kx] as i64;
                sum_r += rv;
                sum_r2 += rv * rv;
            }
        }

        let mut max_sim = f32::NEG_INFINITY;
        let mut max_x: isize = -1;

        for (shift_x, sim_slot) in sim_array.iter_mut().enumerate() {
            // shifted_x: blue パッチ左端の列（x, s_range, k_size, shift_x は全て非負・
            // x が有効範囲内である限り常に画像内に収まることを幾何的に保証済み）
            let shifted_x = x as isize + s_range as isize - k_size as isize - shift_x as isize;
            let blue_col0 = shifted_x as usize;

            let mut sum_b: i64 = 0;
            let mut sum_b2: i64 = 0;
            let mut sum_rb: i64 = 0;
            for ky in 0..kernel_size {
                let red_base = (red_row0 + ky) * w + red_col0;
                let blue_base = (red_row0 + ky) * w + blue_col0;
                for kx in 0..kernel_size {
                    let rv = edge_red[red_base + kx] as i64;
                    let bv = edge_blue[blue_base + kx] as i64;
                    sum_b += bv;
                    sum_b2 += bv * bv;
                    sum_rb += rv * bv;
                }
            }

            let sim = zncc(n, sum_r, sum_r2, sum_b, sum_b2, sum_rb);
            *sim_slot = sim;

            // 端点（shift=0, search_range_x-1）はサブピクセル補間に隣接3点が必要なため除外
            if sim > max_sim && shift_x != 0 && shift_x != search_range_x - 1 {
                max_sim = sim;
                max_x = shift_x as isize;
            }
        }

        if max_x < 1 || max_x > (search_range_x as isize - 2) {
            continue; // 有効ピーク未発見
        }
        let max_x = max_x as usize;
        let sim_neg = sim_array[max_x - 1];
        let sim_peak = sim_array[max_x];
        let sim_pos = sim_array[max_x + 1];

        // strict局所最大でなければタイ・プラトーとして無効化（NaNのまま）
        if !(sim_peak > sim_neg && sim_peak > sim_pos) {
            continue;
        }

        // f32 の評価順を Python と揃える（桁落ちで denominator<=0 になり得るため、
        // このガードは削除できない）
        let denominator = 4.0f32 * sim_peak - 2.0f32 * sim_neg - 2.0f32 * sim_pos;
        let subpixel = if denominator > 0.0 && (sim_pos - sim_neg).abs() <= denominator {
            (sim_pos - sim_neg) / denominator
        } else {
            0.0f32
        };

        let submin = max_x as f32 + subpixel;
        out_row[x] = submin - s_range as f32;
    }
}

/// calcDisparityByZNCC のRust実装本体。GIL解放後に呼ばれる（Python::detach 内）。
#[allow(clippy::too_many_arguments)]
fn calc_disparity_impl(
    red: &[u8],
    blue: &[u8],
    h: usize,
    w: usize,
    kernel_size: usize,
    search_range_x: usize,
    th_value: f64,
    parallel: bool,
) -> Vec<f32> {
    let edge_red = sobel_dx_abs(red, h, w);
    let edge_blue = sobel_dx_abs(blue, h, w);

    let k_size = (kernel_size - 1) / 2;
    let s_range = (search_range_x - 1) / 2;

    // Python の range(a,b)（a>=bなら空）と同じ意味論を符号付き整数で再現する
    let y_start = k_size as isize;
    let y_end = h as isize - k_size as isize;
    let x_start = (s_range + k_size + 1) as isize;
    let x_end = w as isize - (s_range + k_size + 1) as isize;

    let mut out = vec![f32::NAN; h * w];

    // 空画像（h==0/w==0）: 下の (par_)chunks_mut(w) は chunk_size=0 で panic するため先に返す。
    // ラッパー（core/disparity.py）は空入力を ValueError で拒否するので、これは本拡張を
    // 直接呼ぶコードのための防御（画像幅 0 かつ kernel_size=1 で panic していたため）
    if h == 0 || w == 0 {
        return out;
    }

    if y_end <= y_start {
        return out; // 有効な y 範囲なし
    }

    let y_range_start = y_start as usize;
    let y_range_end = y_end as usize;

    let compute_row = |y: usize, out_row: &mut [f32]| {
        compute_disparity_row(
            y,
            out_row,
            &edge_red,
            &edge_blue,
            w,
            kernel_size,
            search_range_x,
            th_value,
            k_size,
            s_range,
            x_start,
            x_end,
        );
    };

    if parallel {
        out.par_chunks_mut(w)
            .enumerate()
            .skip(y_range_start)
            .take(y_range_end - y_range_start)
            .for_each(|(y, out_row)| compute_row(y, out_row));
    } else {
        for (y, out_row) in out
            .chunks_mut(w)
            .enumerate()
            .skip(y_range_start)
            .take(y_range_end - y_range_start)
        {
            compute_row(y, out_row);
        }
    }

    out
}

#[pyfunction]
#[pyo3(signature = (red, blue, kernel_size, search_range_x, th_value, parallel=true))]
fn calc_disparity_by_zncc<'py>(
    py: Python<'py>,
    red: PyReadonlyArray2<'py, u8>,
    blue: PyReadonlyArray2<'py, u8>,
    kernel_size: usize,
    search_range_x: usize,
    th_value: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // 防御的チェック（2D・奇数性は core/disparity.py のラッパーで検証済みという前提だが、
    // 本拡張を直接呼ぶ将来のコードのために再チェックする）
    if kernel_size.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "kernel_size は奇数である必要があります（渡された値: {kernel_size}）"
        )));
    }
    if search_range_x.is_multiple_of(2) {
        return Err(PyValueError::new_err(format!(
            "search_range_x は奇数である必要があります（渡された値: {search_range_x}）"
        )));
    }
    let red_shape = red.shape().to_vec();
    let blue_shape = blue.shape().to_vec();
    if red_shape != blue_shape {
        return Err(PyValueError::new_err(format!(
            "red と blue の形状が一致しません: red={red_shape:?} blue={blue_shape:?}"
        )));
    }
    let h = red_shape[0];
    let w = red_shape[1];

    // GIL保持中に所有バッファへコピー（連続配列を想定・数十µsで完了）。
    // as_slice() は非連続配列だと None を返すため、その場合は to_owned 経由でコピーする。
    let red_vec: Vec<u8> = match red.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => red.as_array().iter().copied().collect(),
    };
    let blue_vec: Vec<u8> = match blue.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => blue.as_array().iter().copied().collect(),
    };

    let result = py.detach(|| {
        calc_disparity_impl(
            &red_vec,
            &blue_vec,
            h,
            w,
            kernel_size,
            search_range_x,
            th_value,
            parallel,
        )
    });

    let arr2 = Array2::from_shape_vec((h, w), result)
        .map_err(|e| PyValueError::new_err(format!("出力配列の形状構築に失敗: {e}")))?;
    Ok(arr2.to_pyarray(py))
}

#[pymodule]
fn _rt_disparity_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(calc_disparity_by_zncc, m)?)?;
    Ok(())
}
