"""NumPy/OpenCV教学练习：无外部数据、无GUI，输出到脚本旁。"""

from pathlib import Path

import cv2
import numpy as np


def project(points_c, K):
    """相机坐标(N,3) → 像素(N,2)；无有效正深度的位置保留NaN。"""
    valid = np.isfinite(points_c).all(axis=1) & (points_c[:, 2] > 1e-6)
    uv = np.full((len(points_c), 2), np.nan)
    q = points_c[valid] @ K.T
    uv[valid] = q[:, :2] / q[:, 2:3]
    return uv, valid


def main():
    out = Path(__file__).resolve().parent / 'practice_output'
    out.mkdir(exist_ok=True)

    # 1. 输入布局与拼接：默认只使用pro+exte。
    x = np.arange(2 * 50 * 441, dtype=np.float32).reshape(2, 50, 441)
    selected = np.concatenate([x[..., :133], x[..., 133:341]], axis=-1)
    assert selected.shape == (2, 50, 341)
    np.testing.assert_array_equal(selected, x[..., :341])
    fric = np.array([0.1, 0.8, 0.3, 0.6], dtype=np.float32)
    stiff = np.array([2., 8., 4., 6.], dtype=np.float32)
    assert np.stack([fric, stiff], axis=1).shape == (4, 2)

    # 2. 视图、归约与图像布局。
    a = np.arange(6).reshape(2, 3)
    view = a[:, :1]
    view[0, 0] = 99
    assert a[0, 0] == 99
    copied = a.copy()
    copied[0, 0] = -1
    assert a[0, 0] == 99
    np.testing.assert_allclose(np.array([[1., 2., 3.], [4., 5., 6.]]).mean(0),
                               [2.5, 3.5, 4.5])

    H, W = 480, 640
    image = np.full((H, W, 3), 45, dtype=np.uint8)
    for u in range(0, W, 40):
        cv2.line(image, (u, 0), (u, H - 1), (65, 65, 65), 1)
    for v in range(0, H, 40):
        cv2.line(image, (0, v), (W - 1, v), (65, 65, 65), 1)
    red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(cv2.cvtColor(red, cv2.COLOR_BGR2RGB)[0, 0],
                                  [255, 0, 0])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255
    assert chw.shape == (3, H, W)

    # 3. T_wc是相机在世界中的位姿；取逆才能将世界点送入相机坐标。
    K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
    T_wc = np.eye(4)
    T_wc[:3, 3] = [1., 0., 0.]
    T_cw = np.linalg.inv(T_wc)
    points_w = np.array([[0.5, -0.3, 2.], [1.5, -0.3, 2.],
                         [0.5, 0.3, 2.], [1.5, 0.3, 2.],
                         [1., 0., -1.], [11., 0., 1.]])
    points_c = points_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    np.testing.assert_allclose(points_c @ T_wc[:3, :3].T + T_wc[:3, 3], points_w)
    uv, positive = project(points_c, K)
    visible = positive & (uv[:, 0] >= 0) & (uv[:, 0] < W)
    visible &= (uv[:, 1] >= 0) & (uv[:, 1] < H)
    np.testing.assert_array_equal(visible, [True, True, True, True, False, False])
    expected = np.array([[220., 180.], [420., 180.], [220., 300.], [420., 300.]])
    np.testing.assert_allclose(uv[:4], expected)
    cv_uv, _ = cv2.projectPoints(points_c[:4], np.zeros(3), np.zeros(3), K, np.zeros(5))
    np.testing.assert_allclose(cv_uv.reshape(-1, 2), expected)

    # 4. 将三维教学方形投影并填入摩擦标签；无监督区域保留NaN。
    label = np.full((H, W), np.nan, dtype=np.float32)
    overlay = image.copy()
    colors = [(255, 100, 30), (30, 80, 255), (80, 220, 100), (30, 220, 220)]
    names = ['LF', 'RF', 'LH', 'RH']
    offsets = np.array([[-0.06, -0.06, 0.], [0.06, -0.06, 0.],
                        [0.06, 0.06, 0.], [-0.06, 0.06, 0.]])
    for i in range(4):
        corners_w = points_w[i] + offsets
        corners_c = corners_w @ T_cw[:3, :3].T + T_cw[:3, 3]
        corners_uv, valid = project(corners_c, K)
        assert valid.all()
        polygon = np.rint(corners_uv).astype(np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        label[mask > 0] = fric[i]
        overlay[mask > 0] = colors[i]
    shown = cv2.addWeighted(image, 0.35, overlay, 0.65, 0)
    for i, (u, v) in enumerate(expected.astype(int)):
        cv2.circle(shown, (int(u), int(v)), 3, (255, 255, 255), -1)
        cv2.putText(shown, f'{names[i]} mu={fric[i]:.1f}', (int(u) - 48, int(v) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        np.testing.assert_allclose(label[v, u], fric[i])
    assert np.isnan(label[0, 0])

    # 5. 缩放/裁剪后的坐标与内参交叉验证。
    A = np.array([[0.5, 0., -10.], [0., 0.5, -20.], [0., 0., 1.]])
    K_new = A @ K
    uv_new, _ = project(points_c[:4], K_new)
    np.testing.assert_allclose(uv_new, expected * 0.5 - [10., 20.])
    resized = cv2.resize(shown, (320, 240), interpolation=cv2.INTER_AREA)
    cropped = resized[20:, 10:].copy()
    assert cropped.shape == (220, 310, 3)

    # 6. 保存与重新加载验证。
    for name, array in [('01_input.png', image), ('02_projected_labels.png', shown),
                        ('03_resized_cropped.png', cropped)]:
        path = out / name
        if not cv2.imwrite(str(path), array):
            raise RuntimeError(f'无法保存: {path}')
        np.testing.assert_array_equal(cv2.imread(str(path)), array)
    np.save(out / 'friction_labels.npy', label)
    np.testing.assert_allclose(np.load(out / 'friction_labels.npy', allow_pickle=False),
                               label, equal_nan=True)
    summary = (f'NumPy {np.__version__}; OpenCV {cv2.__version__}\n'
               f'Observation shape: {selected.shape}\n'
               f'Visible centers: {int(visible.sum())}/6\n'
               f'Pixel centers: {expected.tolist()}\n'
               f'Labeled pixels: {int(np.isfinite(label).sum())}\n'
               f'Label values: {np.unique(label[np.isfinite(label)]).tolist()}\n'
               f'All numerical checks passed. Output: {out}\n')
    (out / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
