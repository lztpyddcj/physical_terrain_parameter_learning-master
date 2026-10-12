# NumPy与OpenCV学习笔记：从数组到机器人视觉投影

适用对象：会Python基本语法，希望阅读机器人视觉、WVN或PTPL代码的学习者。

目标：读懂数组维度；完成图像处理；将三维点投影到图像；区分模型输入、监督标签和几何信息。本笔记是独立教材，示例使用人工数据，不需要机器人、ROS、GPU或外部数据集。

## 学习顺序与使用方法

| 单元 | 内容 | 完成标准 |
|---|---|---|
| 1 | 数组、类型、索引 | 能说明每个轴代表什么 |
| 2 | 维度操作、广播、统计 | 不运行代码也能推断输出形状 |
| 3 | 矩阵与坐标变换 | 能区分逐元素乘法与矩阵乘法 |
| 4 | 图像读写、颜色、缩放、掩码 | 能正确处理RGB/BGR和像素坐标 |
| 5 | 相机投影综合练习 | 能解释三维足端区域怎样对应二维标签 |

每次学习一个单元：先读说明，再手动预测输出，最后运行代码。各Python代码块相互独立，均自行导入所需模块。遇到意外结果，先打印 `shape`、`dtype` 和数值范围。

随附 [practice.py](practice.py) 包含可直接运行的综合练习和数值自检。它是教学示例，不会读取或修改项目数据。

验证记录：已在独立临时环境（Python 3.12、NumPy 2.5.3、OpenCV 5.0.0）运行22个基础代码块和综合脚本，全部通过；已检查生成的投影图。选学的PyTorch转换和真实视频读取两段只进行了语法检查，未验证运行。该验证环境不代表PTPL原项目的兼容环境。

## 0. 准备独立练习环境

在终端执行以下命令；不要把命令粘贴进Python解释器：

```bash
cd /home/ubuntu/桌面/physical_terrain_parameter_learning-master/学习笔记
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python-headless
python -c "import numpy as np, cv2; print(np.__version__, cv2.__version__)"
python practice.py
```

如果提示缺少 `venv` 支持，需要先按系统提示安装对应Python版本的venv组件，或改用已有的独立Conda环境。此环境只做基础练习，不是PTPL训练环境，不需要照搬项目旧版依赖。

`opencv-python-headless` 没有图形窗口支持，适合Docker和无桌面的环境；笔记统一将结果保存成图片查看。如果必须使用 `cv2.imshow`，在另外一个合适环境安装 `opencv-python`。不要在同一个环境同时安装多个OpenCV Python发行包，它们都使用 `cv2` 名称。

运行完成后，查看 `practice_output/` 中的图片和 `summary.txt`。再次运行会覆盖这个练习输出目录中的同名文件。

## 1. NumPy：数组是什么

Python列表适合通用数据；NumPy数组适合规则排列、统一数据类型的数值。图像、机器人观测、位姿矩阵都可以用数组表达。

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(a.shape)   # (2, 3)：2行3列
print(a.ndim)    # 2：有两个轴
print(a.size)    # 6：元素总数
print(a.dtype)   # float32
print(a.nbytes)  # 24：6个元素，每个4字节

print([1, 2] * 2)        # [1, 2, 1, 2]，列表重复
print(np.array([1, 2]) * 2)  # [2 4]，数组逐元素计算
```

### 1.1 shape比变量名更重要

| 形状 | 本练习中的含义 |
|---|---|
| `(4,)` | 四只脚的摩擦值，是一维数组 |
| `(4, 1)` | 四行一列，不等于 `(4,)` |
| `(50, 341)` | 50个时刻，每个时刻341维观测 |
| `(8, 50, 341)` | 一批8段观测序列 |
| `(480, 640, 3)` | 高480、宽640、3通道的图像 |
| `(4, 4)` | 齐次坐标变换矩阵 |

数组只记录尺寸，不会自动知道“哪个轴是时间”。轴的意义由程序约定。

### 1.2 常用创建方式

```python
import numpy as np

print(np.zeros((2, 3), dtype=np.float32))  # 全0
print(np.ones((2, 3), dtype=np.float32))   # 全1
print(np.full((2, 2), 0.5))               # 全0.5
print(np.eye(3))                         # 3×3单位矩阵
print(np.arange(0, 6, 2))                 # [0 2 4]，不含终点
print(np.linspace(0, 1, 5))              # [0. 0.25 0.5 0.75 1.]，默认含终点

rng = np.random.default_rng(42)
observations = rng.normal(size=(8, 50, 441)).astype(np.float32)
print(observations.shape)                 # (8, 50, 441)
```

固定随机种子方便重复练习。`np.empty` 的元素没有初始化，初学时不要把它当作全零数组。

## 2. 索引、切片、掩码与复制

### 2.1 索引取元素，切片取区间

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
# [[ 0, 1, 2, 3],
#  [ 4, 5, 6, 7],
#  [ 8, 9,10,11]]
print(a[1, 2])        # 6
print(a[:, 1])        # [1 5 9]，shape=(3,)
print(a[:, 1:2])      # [[1],[5],[9]]，shape=(3,1)
print(a[-1])          # [8 9 10 11]
print(a[:, ::2])      # 每隔一列取一列
```

切片左闭右开：`1:3` 选择索引1和2。单整数索引通常消去一个轴；切片保留该轴。

在PTPL默认观测布局中：

```python
import numpy as np

x = np.zeros((8, 50, 441), dtype=np.float32)
pro = x[..., :133]        # (8, 50, 133)
exte = x[..., 133:341]    # (8, 50, 208)
hidden = x[..., 341:441]  # (8, 50, 100)
current = x[:, -1, :]     # (8, 441)，每段序列最后一帧
print(pro.shape, exte.shape, hidden.shape, current.shape)
```

`...` 表示省略前面的轴。这里的切片规则来自项目输入约定，不是NumPy对机器人数据的特殊支持。

### 2.2 布尔掩码

```python
import numpy as np

friction = np.array([0.1, 0.8, 0.2, 0.7], dtype=np.float32)
slippery = friction < 0.3
print(slippery)                 # [ True False True False]
print(friction[slippery])       # [0.1 0.2]
print(np.flatnonzero(slippery)) # [0 2]

selected = (friction > 0.05) & (friction < 0.3)
print(selected)
```

数组条件组合使用 `&`、`|`、`~`，每个比较条件加括号；不要用Python的 `and`、`or`。

### 2.3 view与copy：避免意外修改原数据

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = a[:2]             # 基本切片通常返回共享数据的视图
b[0] = 99
print(a)              # [99 2 3 4]

c = a[:2].copy()       # 独立复制
c[0] = -1
print(a[0])           # 99，未被c修改
```

如果要在图像裁剪区域上独立画图，使用 `image[y0:y1, x0:x1].copy()`。布尔/整数数组高级索引通常生成副本，但 `a[mask] = value` 是直接向原数组赋值。

## 3. 维度变化：reshape、transpose、stack

### 3.1 reshape只重新解释排列，transpose交换轴

```python
import numpy as np

a = np.arange(6).reshape(2, 3)
print(a.reshape(3, 2))  # [[0,1],[2,3],[4,5]]
print(a.T)             # [[0,3],[1,4],[2,5]]
print(a.reshape(-1))   # [0 1 2 3 4 5]

image = np.zeros((480, 640, 3), dtype=np.uint8)
chw = image.transpose(2, 0, 1)  # (3,480,640)
batch = chw[None, ...]          # (1,3,480,640)
print(batch.shape)
```

`reshape` 前后元素数量必须相同；`-1` 让NumPy推断一个维度。HWC转CHW必须交换轴，不能用 `reshape(3,H,W)` 冒充。

`squeeze(axis=...)` 删除指定长度为1的轴；不指定axis会删除所有长度为1的轴，batch恰好为1时可能误删批次轴。

### 3.2 concatenate沿已有轴拼接，stack增加新轴

```python
import numpy as np

pro = np.zeros((2, 50, 133), dtype=np.float32)
exte = np.zeros((2, 50, 208), dtype=np.float32)
x = np.concatenate([pro, exte], axis=-1)
print(x.shape)  # (2, 50, 341)

fric = np.array([0.1, 0.2, 0.3, 0.4])
stiff = np.array([2., 3., 4., 5.])
print(np.stack([fric, stiff], axis=0).shape) # (2,4)
print(np.stack([fric, stiff], axis=1).shape) # (4,2)
print(np.concatenate([fric, stiff]).shape)  # (8,)
```

拼接前写下目标形状，比反复试axis可靠。

## 4. 广播、统计与有效数据

### 4.1 广播规则

从最右边的轴开始比较：大小相同，或者其中一个为1，就能兼容；缺少的前导轴按1处理。

```python
import numpy as np

points = np.array([[1., 2., 3.], [4., 5., 6.]]) # (2,3)
offset = np.array([10., 20., 30.])              # (3,)
print(points + offset)  # [[11,22,33],[14,25,36]]

depth = points[:, 2:3]  # (2,1)，保留轴
normalized = points / depth
print(normalized.shape) # (2,3)
```

如果写 `points[:, 2]`，得到 `(2,)`，无法按“每个点除以自己的深度”广播到 `(2,3)`。碰巧点数等于3时甚至可能不报错却算错，所以不能只依赖报错判断。

### 4.2 axis表示沿哪个轴归约

```python
import numpy as np

a = np.array([[1., 2., 3.], [4., 5., 6.]])
print(a.mean())       # 3.5
print(a.mean(axis=0)) # [2.5 3.5 4.5]，沿行方向归约，每列一个结果
print(a.mean(axis=1)) # [2. 5.]，每行一个结果

mean = a.mean(axis=0, keepdims=True) # (1,3)
std = a.std(axis=0, keepdims=True)   # (1,3)，默认ddof=0
z = (a - mean) / (std + 1e-8)
print(z) # [[-1,-1,-1],[1,1,1]]，近似值
```

`keepdims=True` 保留长度为1的轴，便于广播。机器学习中应使用训练数据估计归一化参数，再应用到验证/测试数据，避免信息泄漏。

### 4.3 NaN、掩码与误差

```python
import numpy as np

label = np.array([0.2, np.nan, 0.8])
pred = np.array([0.3, 0.5, 0.6])
valid = np.isfinite(label) & np.isfinite(pred)
if valid.any():
    mae = np.abs(pred[valid] - label[valid]).mean()
    print(mae)  # 约0.15
else:
    print('没有有效标签，不能计算误差')
print(np.isnan(label)) # [False True False]
```

不要用 `label == np.nan`。未知标签不能简单填0，否则会被当作“摩擦确实为0”。PTPL使用NaN标记没有投影监督的像素；有效监督区域与预测置信度区域是两个概念。

## 5. 矩阵运算与坐标系

### 5.1 `*`和`@`

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A * B) # [[5,12],[21,32]]，逐元素乘
print(A @ B) # [[19,22],[43,50]]，矩阵乘
print(np.allclose(np.eye(2) @ A, A)) # True
```

浮点结果一般用 `np.allclose` 比较。解 `Ax=b` 优先使用 `np.linalg.solve(A,b)`，而不是先求逆再乘。

### 5.2 明确变换方向

本笔记约定 `T_cw` 将世界坐标转成相机坐标：`p_c = R_cw @ p_w + t_cw`。相机光学坐标约定X向右、Y向下、Z向前；机器人机身坐标不能直接当作这个坐标系。

```python
import numpy as np

# 示例：两坐标系方向一致，相机中心在世界坐标(1,0,0)
T_wc = np.eye(4)              # 相机坐标 → 世界坐标
T_wc[:3, 3] = [1., 0., 0.]
T_cw = np.linalg.inv(T_wc)    # 世界坐标 → 相机坐标

p_w = np.array([1., 0., 2., 1.]) # 齐次点，最后一项为1
p_c = T_cw @ p_w
print(p_c)                   # [0. 0. 2. 1.]
assert np.allclose(T_wc @ p_c, p_w)
```

变换矩阵的平移列不是在任何方向下都等于“相机位置”。世界中的相机位置出现在 `T_wc`，若方向改为 `T_cw`，旋转和平移都要正确变换。

多个点若按行存储为 `(N,3)`，计算应写成 `points_w @ R_cw.T + t_cw`。这与列向量公式相同，只是存储方式不同。

## 6. OpenCV：图像就是有约定的数组

以下基础示例使用合成图，不需要下载图片。OpenCV读取的普通彩色图像通常是 `uint8`、HWC、BGR。

### 6.1 创建、保存、读取

```python
import cv2
import numpy as np
from pathlib import Path

out = Path('practice_output')
out.mkdir(exist_ok=True)
image = np.zeros((240, 320, 3), dtype=np.uint8)
image[:] = (255, 0, 0)  # BGR，蓝色
assert cv2.imwrite(str(out / 'blue.png'), image)
loaded = cv2.imread(str(out / 'blue.png'), cv2.IMREAD_COLOR)
if loaded is None:
    raise FileNotFoundError('图像未成功读取')
print(loaded.shape, loaded.dtype) # (240,320,3) uint8
```

`imread`失败通常返回None。`imwrite`应检查返回值。路径含空格时，Python字符串可直接使用；终端路径需加引号。

### 6.2 像素索引和绘图坐标不同

```python
import cv2
import numpy as np

image = np.zeros((240, 320, 3), dtype=np.uint8)
image[50, 100] = (0, 0, 255) # 数组索引[y,x]，在(x=100,y=50)设置红色
cv2.circle(image, (100, 50), 8, (0, 255, 0), 2) # 绘图用(x,y)
cv2.rectangle(image, (20, 20), (80, 80), (255, 0, 0), 2)
cv2.putText(image, 'foot', (105, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1)
```

`putText`内置字体适合英文标注，不能假设能正确显示中文。绘图函数直接修改传入图像，需要保留原图时先 `.copy()`。

### 6.3 BGR、RGB、灰度和数值范围

```python
import cv2
import numpy as np

bgr = np.zeros((2, 2, 3), dtype=np.uint8)
bgr[:] = (0, 0, 255) # 红色
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print(rgb[0, 0]) # [255 0 0]
print(gray.shape) # (2,2)

unit = rgb.astype(np.float32) / 255.0
print(unit.min(), unit.max()) # 0.0 1.0
back = np.clip(np.rint(unit * 255.0), 0, 255).astype(np.uint8)
assert np.array_equal(back, rgb)
```

RGB转灰度使用对应的 `COLOR_RGB2GRAY`。颜色顺序由数据约定决定，不是由数组shape决定。保存RGB数组到OpenCV前要转回BGR。

`uint8`不能表示负数或大于255的值，直接算术可能溢出：

```python
import cv2
import numpy as np

a = np.array([[250]], dtype=np.uint8)
b = np.array([[10]], dtype=np.uint8)
print(a + b)           # [[4]]，uint8回绕
print(cv2.add(a, b))   # [[255]]，饱和加法
safe = np.clip(a.astype(np.float32) + 10, 0, 255).astype(np.uint8)
print(safe)            # [[255]]
```

## 7. 裁剪、缩放与相机内参

```python
import cv2
import numpy as np

image = np.zeros((480, 640, 3), dtype=np.uint8)
crop = image[100:300, 200:500].copy()
print(crop.shape) # (200,300,3)
small = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
print(small.shape) # (240,320,3)
```

`resize`的参数是 `(宽,高)`；数组shape是 `(高,宽,通道)`。缩小普通图像可用AREA，放大可用LINEAR；离散类别标签/布尔掩码通常用NEAREST，防止插值产生不存在的类别。

图像变了，投影坐标也要变。假设几何坐标约定是先 `u'=sx*u, v'=sy*v`，再裁去左侧 `x0`、顶部 `y0`：

```text
fx' = sx * fx     fy' = sy * fy
cx' = sx * cx - x0
cy' = sy * cy - y0
```

```python
import numpy as np

K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
sx, sy, x0, y0 = 0.5, 0.5, 10., 20.
A = np.array([[sx, 0., -x0], [0., sy, -y0], [0., 0., 1.]])
K_new = A @ K
print(K_new) # [[200,0,150],[0,200,100],[0,0,1]]
```

这是常用的几何缩放约定。若需要与具体图像采样器达到亚像素级一致，还需核对像素中心定义；例如部分缩放使用 `u'=sx*(u+0.5)-0.5`。不要把不同库的采样约定直接混用。PTPL也会根据缩放和裁剪修正内参。

## 8. 掩码、标签与可视化

掩码是告诉程序“哪些位置参与计算”的数组。`(H,W)`布尔掩码可选出图像中所有对应像素，结果为 `(N,3)`。

```python
import cv2
import numpy as np

image = np.full((240, 320, 3), 80, dtype=np.uint8)
mask_u8 = np.zeros((240, 320), dtype=np.uint8)
cv2.circle(mask_u8, (160, 120), 25, 255, -1)
mask = mask_u8 > 0

label = np.full((240, 320), np.nan, dtype=np.float32)
label[mask] = 0.1
print(np.isfinite(label).sum()) # 被圆覆盖的有效像素数量

overlay = image.copy()
overlay[mask] = (0, 0, 255)
shown = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
assert shown.shape == image.shape
```

这里的红色只是显示颜色，`label`中的浮点数才是监督数据。不能拿JPEG颜色值当作精确摩擦标签。保存数值可用 `.npy`：

```python
import numpy as np
from pathlib import Path

out = Path('practice_output')
out.mkdir(exist_ok=True)
label = np.array([[0.1, np.nan], [0.8, 0.6]], dtype=np.float32)
np.save(out / 'example_label.npy', label)
loaded = np.load(out / 'example_label.npy', allow_pickle=False)
assert np.allclose(label, loaded, equal_nan=True)
```

## 9. 综合：三维点怎样变成图像标签

### 9.1 投影公式

对相机坐标中的点 `(X,Y,Z)`，忽略畸变时：

```text
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

`fx,fy`为以像素表示的焦距，`cx,cy`为主点。必须先检查 `Z>0`，再检查 `0<=u<W`、`0<=v<H`。相机后方或图像外的点不能写入标签图。

```python
import numpy as np

K = np.array([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]])
points_c = np.array([[0., 0., 2.], [0.5, 0., 2.], [0., 0.5, 2.]])
homogeneous = points_c @ K.T
uv = homogeneous[:, :2] / homogeneous[:, 2:3]
print(uv) # [[320,240],[420,240],[320,340]]
```

点深度越大，相同横向位移产生的像素位移越小。把第二个点的Z从2改成4，观察u由420变成370。

### 9.2 完整练习脚本

运行随附 `practice.py`：它创建四个假想接触区域，执行世界→相机变换，投影到合成图像，并赋予示例摩擦标签。再加入相机后方和画面外的点，验证过滤逻辑。

预期四个中心像素为：

| 接触区域 | 图像中心 `(u,v)` | 教学用摩擦标签 |
|---|---|---:|
| LF | `(220,180)` | 0.1 |
| RF | `(420,180)` | 0.8 |
| LH | `(220,300)` | 0.3 |
| RH | `(420,300)` | 0.6 |

脚本还会使用 `cv2.projectPoints` 与NumPy公式交叉检查，以及验证缩放/裁剪后的内参。脚本中的区域是平行于成像平面的教学方形，不代表真实足底几何；标签是人为指定，不是模型预测。

真实PTPL还需要足端接触过滤、时间关联、历史相机位姿和区域渲染等步骤。几何投影落在画面内并不保证没有遮挡；二维点投影本身也不能恢复物理摩擦值。摩擦标签必须来自另一个监督来源。

### 9.3 镜头畸变与标定：先理解，后实做

实际镜头可能有径向/切向畸变。`cv2.projectPoints`可结合畸变系数投影；`cv2.undistort`可校正图像。相机标定用于从已知结构的多张图像估计内参与畸变，不是随便填一个K矩阵。

对于已经去畸变的图像，应使用对应的新内参和一致的投影模型，不能再次重复应用原畸变。综合练习故意使用零畸变，以便先理解几何关系。

## 10. 选学：与PyTorch对接

本节需要另行安装适合自己环境的PyTorch；基础练习不依赖它。

```python
import numpy as np
import torch

rgb = np.zeros((240, 320, 3), dtype=np.uint8)
chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
x = torch.from_numpy(chw).float().div(255.0).unsqueeze(0)
print(x.shape) # torch.Size([1,3,240,320])

back = x.detach().cpu().numpy()
print(back.shape) # (1,3,240,320)，转换不会自动改布局
```

`torch.from_numpy`在支持的情况下与原数组共享CPU内存；不希望共享时复制。`detach()`表示脱离梯度记录，`.cpu()`将张量转到CPU。颜色通道反转切片可能产生负步长，`np.ascontiguousarray`可避免相关转换问题。实际模型可能还要求均值/标准差归一化；是否在模型内已经执行，要查接口，不能重复归一化。

## 11. 选学：视频逐帧处理

准备真实视频文件后再运行：

```python
import cv2

cap = cv2.VideoCapture('input.mp4') # 修改为实际路径
if not cap.isOpened():
    raise RuntimeError('无法打开视频')
count = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
finally:
    cap.release()
print('读取帧数:', count)
```

保存视频时，`VideoWriter`需要帧率和 `(W,H)`，每帧尺寸必须一致，并应检查 `isOpened()`、最终调用 `release()`。编码器是否可用取决于环境。ROS bag并不是普通MP4，需用ROS bag接口读取消息后再转换图像。

## 12. 自测题

先自己作答，再看后面的答案。

1. `x.shape=(8,50,441)`，取最后一帧且保留时间轴，怎么写？
2. `image.shape=(480,640,3)`，取左上角高100、宽200的区域，shape是什么？
3. `(N,384)`特征矩阵，对每个特征维度求平均应该用哪个axis？
4. 四足摩擦和刚度均为 `(4,)`，如何组成每行一只脚的 `(4,2)`？
5. 对 `(N,3)` 的相机点，每个点除以自己的深度，深度应取什么shape？
6. 为什么 `image.reshape(3,480,640)` 不能正确替代HWC→CHW？
7. K中的 `fx=400,cx=320`，点 `(X,Y,Z)=(0.5,0,2)` 的u是多少？若Z变成4呢？
8. 未标注区域为什么使用NaN，而不是填0？
9. 图像缩小到一半，随后左侧裁10像素，`cx=320`如何变化（采用第7节约定）？
10. 如果监督投影错了位置，神经网络损失是否仍可能下降？这说明什么？

### 参考答案

1. `x[:, -1:, :]`，shape为 `(8,1,441)`。
2. `image[:100, :200]`，shape为 `(100,200,3)`；需独立修改时加 `.copy()`。
3. `features.mean(axis=0)`，得到 `(384,)`。
4. `np.stack([fric, stiff], axis=1)`。
5. `points[:, 2:3]`，shape为 `(N,1)`；先筛选有效正深度。
6. reshape重新安排尺寸，不交换通道轴的语义；应 `transpose(2,0,1)`。
7. 420；370。
8. 0也是有效摩擦值；未知与真实为0必须区分。
9. `320*0.5-10=150`。
10. 可能。网络可以拟合错误监督，损失下降不能单独证明投影与物理含义正确。

## 13. 常见问题速查

| 现象 | 优先检查 |
|---|---|
| `NoneType`没有shape | `imread`是否失败、路径是否正确 |
| 图像颜色不对 | BGR/RGB是否混淆 |
| 保存图像几乎全黑 | 是否把0–1浮点值直接当0–255图像保存 |
| 图像宽高颠倒 | `resize`使用(W,H)，shape使用(H,W,C) |
| 标签与图像错位 | 变换方向、相机内参、缩放/裁剪、畸变和时间同步 |
| 数值正常但空间关系不对 | 是否把reshape当transpose、是否交换x/y |
| 算术结果突然变小 | uint8溢出，先转浮点再计算 |
| 广播错误或悄悄算错 | 每个轴的含义、是否需要保留长度1的轴 |
| 原图被意外修改 | 基本切片共享内存，是否需要copy |
| 平均误差为NaN | 是否过滤无标签区域，以及是否存在有效样本 |

## 14. 学完后回到项目看什么

1. `physical_decoder/physical_decoder/decoder_config.py`：把输入切片与第2节对应。
2. `physical_decoder_training/training_utils/data_manager.py`：观察序列截取、拼接、标签选择。
3. `base_wvn/base_wvn/utils/feat_extractor.py`：看图像缩放、裁剪与尺寸约定。
4. `base_wvn/src/wild_visual_navigation_ros/scripts/ros_converter.py`：看图像/位姿转换与内参更新。
5. `base_wvn/base_wvn/utils/image_projector.py`：看三维区域投影。
6. `base_wvn/base_wvn/graph_manager/manager.py`：看监督区域如何与物理标签结合。

不要求背所有函数。能解释形状、颜色、坐标、有效区域和数据来源，才是这一阶段的学习目标。

## 官方查阅入口

以下为API查阅资料；本笔记的数值示例和综合实验独立编写。

- [NumPy基础](https://numpy.org/doc/stable/user/absolute_beginners.html)：查询数组属性、创建和索引。
- [NumPy广播规则](https://numpy.org/doc/stable/user/basics.broadcasting.html)：查询维度兼容条件。
- [OpenCV基本图像操作](https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html)：查询图像数组操作。
- [OpenCV相机标定](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)：查询畸变、标定和投影接口。
