# Train Mask Annotator

一个独立、只读原始 episode 的测试标注网页。人工框选的红色区间会导出为逐帧布尔 `train_mask`：红色为 `False`，其他为 `True`。

默认测试数据：

```text
/media/yns/T9/0824_yns_unscrew_cap/2026-08-24_20-45-49.080158
```

## 启动

```bash
cd /home/yns/deploy/Deploy-v2/train_mask_annotator
python3 server.py
```

浏览器打开 <http://127.0.0.1:8765>。

第一次启动会在 `/tmp/train-mask-annotator` 缓存浏览器兼容的 H.264 视频，不修改 T9 上的原始文件。后续启动会复用缓存。也可以使用 `--no-browser-proxy` 直接提供原始 MJPEG MP4。

指定其他 episode：

```bash
python3 server.py --episode /path/to/episode --port 8765
```

## 标注

- `Space`：播放/暂停
- `←` / `→`：前后 1 帧
- `Shift + ←` / `Shift + →`：前后 1 秒
- `[`：错误动作开始
- `]`：有效恢复开始；生成 `[错误开始, 恢复开始)` 红色区间
- `Ctrl+S`：保存
- `Esc`：取消尚未完成的区间

一条 episode 可以添加多个红色区间。重叠区间会自动合并。

## 输出

结果写在 `annotations/`：

- `<episode>.json`：区间和时间信息，便于复查与修改
- `<episode>.train_mask.npy`：shape `[T]`、dtype `bool` 的逐帧 mask

工具只根据人工红色区间生成 mask，不修改原始 episode，也不在这里绑定 observation history 或 action horizon。

## 测试

```bash
cd /home/yns/deploy/Deploy-v2/train_mask_annotator
python3 -m unittest discover -s tests -v
node --check static/app.js
```
