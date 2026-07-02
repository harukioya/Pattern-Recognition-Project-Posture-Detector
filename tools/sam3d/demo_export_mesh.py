# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
import time
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample_together
from tqdm import tqdm


def save_obj(path, vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Exported from SAM 3D Body\n")
        file.write(f"# vertices: {len(vertices)}\n")
        file.write(f"# faces: {len(faces)}\n")

        for vertex in vertices:
            file.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")

        for face in faces:
            # OBJ indices are 1-based.
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def get_mesh_vertices(person_output, mesh_space):
    vertices = np.asarray(person_output["pred_vertices"], dtype=np.float32)
    if mesh_space == "camera":
        vertices = vertices + np.asarray(person_output["pred_cam_t"], dtype=np.float32)
    return vertices


def to_viewer_vertices(vertices):
    viewer_vertices = np.asarray(vertices, dtype=np.float32).copy()
    viewer_vertices[:, 1] *= -1
    viewer_vertices[:, 2] *= -1
    return viewer_vertices


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    total_start = time.perf_counter()
    init_start = time.perf_counter()

    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[timing] device: {device}")

    model_start = time.perf_counter()
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )
    sync_cuda(device)
    print(f"[timing] SAM 3D Body model load: {time.perf_counter() - model_start:.2f}s")

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )

    if (
        args.segmentor_name == "sam2"
        and len(segmentor_path)
        or args.segmentor_name != "sam2"
    ):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    sync_cuda(device)
    print(f"[timing] total initialization: {time.perf_counter() - init_start:.2f}s")

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list, desc="images"):
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        image_start = time.perf_counter()

        sync_cuda(device)
        inference_start = time.perf_counter()
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )
        sync_cuda(device)
        inference_seconds = time.perf_counter() - inference_start

        if len(outputs) == 0:
            print(
                f"[timing] {image_base}: people=0 "
                f"inference={inference_seconds:.2f}s total={time.perf_counter() - image_start:.2f}s"
            )
            print(f"No humans detected in {image_path}")
            continue

        render_start = time.perf_counter()
        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        render_path = os.path.join(output_folder, f"{image_base}.png")
        cv2.imwrite(render_path, rend_img.astype(np.uint8))
        render_seconds = time.perf_counter() - render_start

        mesh_start = time.perf_counter()
        for person_idx, person_output in enumerate(outputs):
            vertices = get_mesh_vertices(person_output, args.mesh_space)
            mesh_path = os.path.join(
                output_folder, f"{image_base}_person{person_idx}.obj"
            )
            save_obj(mesh_path, vertices, estimator.faces)
            if args.export_viewer_obj:
                viewer_mesh_path = os.path.join(
                    output_folder, f"{image_base}_person{person_idx}_viewer.obj"
                )
                save_obj(viewer_mesh_path, to_viewer_vertices(vertices), estimator.faces)
        mesh_seconds = time.perf_counter() - mesh_start

        print(
            f"[timing] {image_base}: people={len(outputs)} "
            f"inference={inference_seconds:.2f}s render={render_seconds:.2f}s "
            f"mesh_export={mesh_seconds:.2f}s total={time.perf_counter() - image_start:.2f}s"
        )

    print(f"[timing] total run: {time.perf_counter() - total_start:.2f}s for {len(images_list)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - export rendered images and OBJ meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo_export_mesh.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_MHR_PATH)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--mesh_space",
        choices=["model", "camera"],
        default="model",
        help="OBJ coordinate space. Use `model` for centered meshes, `camera` to include camera translation.",
    )
    parser.add_argument(
        "--export_viewer_obj",
        action="store_true",
        default=False,
        help="Also export a viewer-oriented OBJ with the same 180-degree X-axis correction used by the renderer.",
    )
    args = parser.parse_args()

    main(args)

