"""Video reconstruction interface."""

import argparse
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache/uploads"))


def build_app(config=None, device=None, precision=None):
    import gradio as gr

    from runtime.pipeline import Predictor, video_plan

    predictor = Predictor(config, device, precision=precision)

    def layout(video, policy):
        if not video:
            return "上传视频后显示处理尺寸。"
        try:
            plan = video_plan(video, policy)
            source = plan["source_size"]
            target = plan["size"]
            return f"{source[0]}×{source[1]} → {target[0]}×{target[1]}；叠加视频保持原图尺寸。"
        except (OSError, ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    def status(result):
        m = result["summary"]
        t = m["timings"]
        text = f"完成 {m['frames']} 帧 · 推理 {t['inference_seconds']:.2f}s · {t['fps']:.2f} FPS"
        details = [
            f"{label} {t[name]:.2f}s"
            for name, label in (
                ("load_seconds", "加载"),
                ("prepare_warmup_seconds", "准备"),
                ("render_seconds", "绘制"),
                ("total_seconds", "总耗时"),
            )
            if name in t
        ]
        camera_source = (
            "指定内参" if m.get("geometry_camera_source") == "calibration" else "模型预测"
        )
        return text + "\n\n" + " · ".join(details) + f" · 推理相机：{camera_source}"

    def predict(
        video,
        policy,
        mode,
        threshold,
        limit,
        camera,
        correction,
        geometry_camera=None,
        progress=gr.Progress(),
    ):
        if not video:
            raise gr.Error("请先上传视频")
        folder = ROOT / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            result = predictor.run(
                video,
                folder,
                resize_mode=policy,
                overlay_mode=mode,
                threshold=threshold,
                max_frames=int(limit),
                intrinsics=camera,
                geometry_intrinsics=geometry_camera,
                global_correction=correction,
                progress=lambda value, text: progress(value, desc=text),
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        return (
            result["overlay"],
            [result["prediction"], result["metadata"], result["overlay"]],
            status(result),
            result["raw_directory"],
        )

    def redraw(video, folder, mode, threshold, camera, correction, progress=gr.Progress()):
        if not folder or not video:
            raise gr.Error("请先运行推理")
        try:
            result = predictor.rerender(
                video,
                folder,
                overlay_mode=mode,
                threshold=threshold,
                intrinsics=camera,
                global_correction=correction,
                progress=lambda value, text: progress(value, desc=text),
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error
        return (
            result["overlay"],
            [result["prediction"], result["metadata"], result["overlay"]],
            "叠加视频已更新。",
        )

    def unload():
        predictor.unload()
        return "模型显存已释放。"

    with gr.Blocks(title="视频重建") as demo:
        gr.Markdown("# 视频重建\n上传视频，生成相机坐标系中的三维运动和叠加预览。")
        with gr.Row():
            with gr.Column():
                video = gr.File(
                    label="视频",
                    file_types=[".mp4", ".mov", ".MOV", ".mkv", ".avi", ".webm", ".m4v"],
                    type="filepath",
                )
                preview = gr.Video(label="输入预览", format="mp4", interactive=False)
            output = gr.Video(label="重建结果", interactive=False)
        with gr.Row():
            policy = gr.Radio(
                choices=[("中心裁剪", "center_crop"), ("保留完整宽屏画面", "long_edge_pad")],
                value="center_crop",
                label="尺寸策略",
            )
            mode = gr.Dropdown(
                choices=[("关节", "joints"), ("网格", "mesh"), ("二维关键点", "anchors")],
                value="joints",
                label="叠加方式",
            )
            threshold = gr.Slider(0, 1, value=0.5, step=0.05, label="显示阈值")
        size_info = gr.Markdown("上传视频后显示处理尺寸。")
        with gr.Accordion("更多设置", open=False):
            limit = gr.Number(value=0, minimum=0, precision=0, label="最大帧数（0 为全部）")
            geometry_camera = gr.Textbox(
                label="推理相机内参 K",
                placeholder="[[fx,0,cx],[0,fy,cy],[0,0,1]]",
                info="原视频像素坐标中的去畸变针孔内参。留空使用模型预测；修改后需重新运行推理，不能仅重绘。",
            )
            camera = gr.Textbox(
                label="叠加显示内参 K",
                placeholder="[[fx,0,cx],[0,fy,cy],[0,0,1]]",
                info="可选，仅改变叠加显示，不改变三维预测。留空使用推理结果中的相机。",
            )
            correction = gr.Checkbox(
                value=False,
                label="全局一致化",
                info="统一整段视频的手形和相机；指定推理 K 时固定相机。原始结果保留。启用前请清空叠加显示内参。",
            )
        with gr.Row():
            run = gr.Button("运行推理", variant="primary")
            redraw_button = gr.Button("仅更新叠加视频")
            release = gr.Button("释放显存")
        info = gr.Markdown("准备就绪。首次推理包含模型加载和编译预热。")
        downloads = gr.File(label="下载结果", file_count="multiple", interactive=False)
        state = gr.State(None)
        video.change(lambda path: path, video, preview, queue=False, api_name=False)
        video.change(layout, [video, policy], size_info, queue=False, api_name=False)
        video.change(lambda: None, outputs=state, queue=False, api_name=False)
        policy.change(layout, [video, policy], size_info, queue=False, api_name=False)
        run.click(
            predict,
            [video, policy, mode, threshold, limit, camera, correction, geometry_camera],
            [output, downloads, info, state],
            concurrency_id="model",
            concurrency_limit=1,
            api_name="predict",
        )
        redraw_button.click(
            redraw,
            [video, state, mode, threshold, camera, correction],
            [output, downloads, info],
            concurrency_id="model",
            concurrency_limit=1,
            api_name="rerender",
        )
        release.click(
            unload, outputs=info, concurrency_id="model", concurrency_limit=1, api_name="unload"
        )
    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp8"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    (ROOT / "outputs").mkdir(exist_ok=True)
    build_app(args.config, args.device, args.precision).launch(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=[str(ROOT / "outputs")],
        show_error=True,
    )


if __name__ == "__main__":
    main()
