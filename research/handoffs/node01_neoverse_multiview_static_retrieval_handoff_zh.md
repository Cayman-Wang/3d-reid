# NeoVerse 三机联合静态 retrieval 分支执行交接

## 修改文件清单

- `mvp-demo/scripts/render_neoverse_multiview_preview.py`
- `mvp-demo/README.md`
- `research/guides/node01_neoverse_multiview_static_retrieval_zh.md`

## 实际执行命令清单

```powershell
D:\ML\anaconda3\envs\neoverse\python.exe mvp-demo/scripts/render_neoverse_multiview_preview.py --help
```

## 生成的关键产物路径

- `mvp-demo/scripts/render_neoverse_multiview_preview.py`

## 已知限制与未解决问题

- 新增的轻量预览脚本只做 rasterization，不加载 diffusion / T5 / VAE / LoRA；默认依赖同目录 `probe_meta.json` 读取分辨率与 resize 模式。
- 预览脚本默认输出原视角回放、对比视频和 orbit 预览；如果 bundle 没有对应字段，会直接报错，不做静默回退。
