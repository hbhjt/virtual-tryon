# 智能柜 CPU 虚拟换衣

这是一个面向无独立 GPU Windows 设备的本地虚拟换衣项目。网页接收摄像头视频流或照片，从视频流中选择姿态稳定、清晰度较高的一帧，输出人物试穿服装的照片。

项目提供两档效果：

- 快速预览：MediaPipe 姿态、服装整图等比缩放/平移/轻微旋转、边缘与光照融合，典型耗时约 0.4 秒。快速模式不会对衣服做局部网格拉伸，也不会使用人体分割或手臂蒙版裁切服装轮廓。
- AI 高质量：SCHP 原衣解析 + CatVTON 扩散式虚拟试衣，512×768、默认 12 步，当前测试机约 3–5 分钟。

## 已完成的六个阶段

1. 4 件真实纹理透明服装：T 恤、Polo、长袖针织衫、轻薄夹克。
2. 短袖和长袖分类模板；服装元数据记录版型、松量、衣长和专用锚点。
3. 快速模式使用整件衣服的刚性相似变换，保持领口、肩线、袖型和下摆之间的原始比例；分类网格仅供 AI 生成蒙版使用。
4. 基于人体姿态、肤色和部位区域的 CPU 遮挡解析，手臂、手、颈部、下装可回到服装前方。
5. 局部光照场、材质细节增强、接触阴影和 1～2 像素抗锯齿边缘。
6. CatVTON CPU 高质量模式，独立 Python 环境运行，不影响快速模式。

## 快速启动

在项目目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run.ps1
```

浏览器访问 <http://127.0.0.1:8000>。以后启动可使用：

```powershell
.\run.ps1 -SkipInstall
```

## 安装 AI 高质量模式

AI 环境与网页主环境隔离：

```powershell
.\scripts\setup_ai.ps1
```

首次在网页选择“AI 高质量”会下载约 5GB 模型，缓存到 `models/catvton-cache/`，之后可离线复用。默认精细档为 512×768、12 步，16GB 内存的纯 CPU 设备预计约 3–5 分钟。

CatVTON 源码固定在提交 `3b795364a4d2f3b5adb365f39cdea376d20bc53c`。其模型和代码采用非商业许可，商业智能柜落地前必须重新核对并取得适用授权。

## 推荐拍摄方式

- 单人正面站立，露出头部、双肩、手臂和腰部。
- 摄像头固定，光线均匀，避免强逆光。
- 双臂自然下垂时快速模式最稳定；交叉手臂、侧身和长发遮挡优先使用 AI 高质量模式。
- 两种模式都恢复为输入图片的原始尺寸和比例；AI 模式内部使用 512×768 等比留白推理。

## 服装素材

内置素材位于 `garments/`。每个目录包含：

- `image.png`：透明背景成品。
- `source-chroma.png`：生成时的纯色背景源图。
- `metadata.json`：品类模板、松量、衣长与锚点。

导入自定义服装时，推荐正面平铺、透明 PNG、无衣架/模特/文字水印，分辨率不低于 600×700。普通 JPG 会尝试自动抠图，并按短袖上衣模板处理。

## 示例与测试

- 测试人物：`samples/inputs/`
- 快速效果对比：`samples/outputs/comparison-woman.png`、`comparison-man.png`
- 已验证的 AI 输出：`samples/outputs/ai-woman-coral-tee.jpg`

重新生成快速示例：

```powershell
.\.venv\Scripts\python.exe scripts\generate_samples.py
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 主要接口

- `GET /api/health`：姿态、服装和 AI 后端状态。
- `GET /api/garments`：服装目录。
- `POST /api/garments`：导入服装，字段 `name`、`image`。
- `POST /api/analyze`：分析视频帧，字段 `image`。
- `POST /api/tryon`：字段 `garment_id`、`mode=fast|ai`、`image`。
- `POST /api/tryon/burst`：字段 `garment_id`、`mode=fast|ai`、1～5 个 `images`。
- `GET /docs`：交互式接口文档。

## 项目结构

```text
app/
  main.py          FastAPI 服务与双模式接口
  pose.py          MediaPipe 姿态与画面评分
  garments.py      服装目录、导入和模板元数据
  tryon.py         快速保形定位、AI 蒙版网格、光影和合成
  human_parser.py  CPU 人体部位遮挡解析
  ai_tryon.py      CatVTON 独立进程封装
scripts/
  setup_ai.ps1
  catvton_worker.py
garments/          透明服装与锚点元数据
models/            姿态模型和 AI 模型缓存
samples/           输入与验收输出
tests/             核心算法和 API 测试
```

## 已知边界

快速模式不会重新生成被原衣服遮挡的皮肤和头发，因此领口差异很大、复杂交叉遮挡或侧身时可能残留原衣服；这类输入应使用 AI 高质量模式。AI 模式更自然，但 CPU 耗时明显，且输出细节会受随机种子、遮罩和输入服装构图影响，不能用于判断真实尺码。
