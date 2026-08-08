# 虚拟换衣示例素材

`inputs/` 中的两张图片是为本项目生成的正面测试人像，可以直接在智能试衣页面点击“上传照片”进行测试：

- `person-woman-front.png`
- `person-man-front.png`

`outputs/` 包含项目当前 CPU 几何换装管线的实际输出：

- `comparison-woman.png`：女性测试图的原图与三件默认上衣对比；
- `comparison-man.png`：男性测试图的原图与三件默认上衣对比；
- 独立 JPG：每个人物与每件上衣的单独结果；
- `manifest.json`：姿态检测、质量评分和合成耗时。

重新生成全部结果：

```powershell
.\.venv\Scripts\python.exe scripts\generate_samples.py
```

这些示例图是 AI 生成的虚构人物，不对应真实个人。第一阶段采用几何变形与图层合成，示例用于验证上衣位置、人物体型适配和处理速度，不代表扩散生成模型的照片级布料重绘效果。

