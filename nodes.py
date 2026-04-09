import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import logging
import copy
import datetime
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .vitpose_utils.utils import bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq, aaposemeta_to_dwpose_scail

def convert_openpose_to_target_format(frames, max_people=2):
    NUM_BODY = 18
    NUM_FACE = 70
    NUM_HAND = 21

    results = []
    for frame in frames:
        canvas_width = frame['canvas_width']
        canvas_height = frame['canvas_height']
        people = frame['people'][:max_people]

        bodies = []
        hands = []
        faces = []
        body_scores = []
        hand_scores = []
        face_scores = []

        for person in people:
            pose_raw = person.get('pose_keypoints_2d') or []
            if len(pose_raw) != NUM_BODY * 3:
                continue

            pose = np.array(pose_raw).reshape(-1, 3)
            pose_xy = np.stack([pose[:, 0] / canvas_width, pose[:, 1] / canvas_height], axis=1)
            bodies.append(pose_xy)
            body_scores.append(pose[:, 2])

            face_raw = person.get('face_keypoints_2d') or []
            if len(face_raw) == NUM_FACE * 3:
                face = np.array(face_raw).reshape(-1, 3)
                face_xy = np.stack([face[:, 0] / canvas_width, face[:, 1] / canvas_height], axis=1)
                faces.append(face_xy)
                face_scores.append(face[:, 2])

            hand_left_raw = person.get('hand_left_keypoints_2d') or []
            hand_right_raw = person.get('hand_right_keypoints_2d') or []
            if len(hand_left_raw) == NUM_HAND * 3:
                hand_left = np.array(hand_left_raw).reshape(-1, 3)
                hand_left_xy = np.stack([hand_left[:, 0] / canvas_width, hand_left[:, 1] / canvas_height], axis=1)
                hands.append(hand_left_xy)
                hand_scores.append(hand_left[:, 2])
            if len(hand_right_raw) == NUM_HAND * 3:
                hand_right = np.array(hand_right_raw).reshape(-1, 3)
                hand_right_xy = np.stack([hand_right[:, 0] / canvas_width, hand_right[:, 1] / canvas_height], axis=1)
                hands.append(hand_right_xy)
                hand_scores.append(hand_right[:, 2])

        result = {
            'bodies': {
                'candidate': np.array(bodies, dtype=np.float32),
                'subset': np.array([np.arange(NUM_BODY) for _ in bodies], dtype=np.float32) if bodies else np.array([])
            },
            'hands': np.array(hands, dtype=np.float32),
            'faces': np.array(faces, dtype=np.float32),
            'body_score': np.array(body_scores, dtype=np.float32),
            'hand_score': np.array(hand_scores, dtype=np.float32),
            'face_score': np.array(face_scores, dtype=np.float32)
        }
        results.append(result)
    return results

def scale_faces(poses, pose_2d_ref):
    # Input: two lists of dict, poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
    # Scale the facial keypoints in poses according to the center point of the face
    # That is: calculate the distance from the center point (idx: 30) to other facial keypoints in ref,
    # and the same for poses, then get scale_n as the ratio
    # Clamp scale_n to the range 0.8-1.5, then apply it to poses
    # Note: poses are modified in place

    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    # Extract numpy arrays
    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    # Center point (nose tip or face center)
    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    # Calculate distance to center point
    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    # Avoid the 0 distance of the center point itself
    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    # Clamp to [0.8, 1.5]
    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        # Extract numpy array
        face = np.array(face[0])      # (68, 2)
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])   # (14, 2)
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    # In-place modification
    pose['faces'][0] = scaled_face

    return scale_n

def merge_dwpose_results(person_dwposes):
    """Merge multiple single-person DWPose dicts into one multi-person dict."""
    if len(person_dwposes) == 1:
        return person_dwposes[0]
    return {
        "bodies": {
            "candidate": np.concatenate([p["bodies"]["candidate"] for p in person_dwposes], axis=0),
            "subset": np.concatenate([p["bodies"]["subset"] for p in person_dwposes], axis=0),
        },
        "hands": np.concatenate([p["hands"] for p in person_dwposes], axis=0),
        "faces": np.concatenate([p["faces"] for p in person_dwposes], axis=0),
        "body_score": np.concatenate([p["body_score"] for p in person_dwposes], axis=0),
        "hand_score": np.concatenate([p["hand_score"] for p in person_dwposes], axis=0),
        "face_score": np.concatenate([p["face_score"] for p in person_dwposes], axis=0),
    }

class PoseDetectionVitPoseToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": ("POSEMODEL",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "ViTPose to DWPose format pose detection node."

    def process(self, vitpose_model, images):

        detector = vitpose_model["yolo"]
        pose_model = vitpose_model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B*2)
        progress = 0

        bboxes_per_frame = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape,
                single_person=False
                )[0]
            frame_bboxes = []
            for det in detections:
                bbox = det["bbox"]
                if bbox is not None and bbox[-1] > 0 and (bbox[2] - bbox[0]) >= 10 and (bbox[3] - bbox[1]) >= 10:
                    frame_bboxes.append(bbox)
            if not frame_bboxes:
                frame_bboxes = [np.array([0, 0, W, H])]
            bboxes_per_frame.append(frame_bboxes)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        dwposes = []
        for img, frame_bboxes in tqdm(zip(images_np, bboxes_per_frame), total=len(images_np), desc="Extracting keypoints"):
            person_dwposes = []
            for bbox in frame_bboxes:
                center, scale = bbox_from_detector(bbox, input_resolution, rescale=rescale)
                cropped = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

                img_norm = (cropped - IMG_NORM_MEAN) / IMG_NORM_STD
                img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

                keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
                meta = load_pose_metas_from_kp2ds_seq(keypoints, width=W, height=H)[0]
                person_dwposes.append(aaposemeta_to_dwpose_scail(meta))

            dwposes.append(merge_dwpose_results(person_dwposes))
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        swap_hands = True
        out_dict = {"poses": dwposes, "swap_hands": swap_hands}
        return out_dict,


class ConvertOpenPoseKeypointsToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "max_people": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of people to process per frame"}),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Convert OpenPose format keypoints to DWPose format."

    def process(self, keypoints, max_people=2):
        swap_hands = False
        out_dict = {"poses": convert_openpose_to_target_format(keypoints, max_people=max_people), "swap_hands": swap_hands}
        return out_dict,


def filter_to_single_person(pose_input, dw_pose_input, intrinsic_matrix, height, width, debug_lines=None):
    """Filter multi-person NLF and DWPose inputs to the main character only.

    Main character = largest 3D body extent in first valid frame, tracked by pelvis proximity.
    DWPose person is matched by projecting the NLF main person's head joint to 2D.
    """
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    if debug_lines is not None:
        debug_lines.append("=== FILTER_TO_SINGLE_PERSON ===")
        debug_lines.append(f"intrinsic_matrix: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        debug_lines.append(f"image size: {width}x{height}")
        debug_lines.append(f"total NLF frames: {len(pose_input)}")
        debug_lines.append(f"total DW frames: {len(dw_pose_input) if dw_pose_input else 'None'}")
        debug_lines.append("")

    main_idx = 0
    for frame_poses in pose_input:
        if frame_poses.shape[0] == 0:
            continue

        if debug_lines is not None:
            debug_lines.append(f"--- Person Selection (first valid frame) ---")
            debug_lines.append(f"NLF people detected: {frame_poses.shape[0]}")

        max_extent = -1
        for p_idx in range(frame_poses.shape[0]):
            person = frame_poses[p_idx]
            person_np = person.cpu().numpy() if isinstance(person, torch.Tensor) else person

            if debug_lines is not None:
                pelvis = person_np[0]
                mean_z = np.mean(person_np[:, 2])
                extent_3d = np.sum(np.max(person_np, axis=0) - np.min(person_np, axis=0))
                valid_z = person_np[:, 2] > 0.01
                if np.any(valid_z):
                    pts = person_np[valid_z]
                    u_pts = (fx * pts[:, 0] / pts[:, 2] + cx) / width
                    v_pts = (fy * pts[:, 1] / pts[:, 2] + cy) / height
                    area_2d = (np.max(u_pts) - np.min(u_pts)) * (np.max(v_pts) - np.min(v_pts))
                    u_range = f"[{np.min(u_pts):.4f}, {np.max(u_pts):.4f}]"
                    v_range = f"[{np.min(v_pts):.4f}, {np.max(v_pts):.4f}]"
                else:
                    area_2d = 0.0
                    u_range = "N/A"
                    v_range = "N/A"
                debug_lines.append(
                    f"  Person {p_idx}: pelvis=({pelvis[0]:.1f}, {pelvis[1]:.1f}, {pelvis[2]:.1f}), "
                    f"mean_Z={mean_z:.1f}, 3D_extent={extent_3d:.1f}, "
                    f"2D_area={area_2d:.6f}, u_range={u_range}, v_range={v_range}"
                )

            if np.sum(np.abs(person_np)) < 0.01:
                if debug_lines is not None:
                    debug_lines.append(f"    -> SKIPPED (near-zero joints)")
                continue
            extent = np.sum(np.max(person_np, axis=0) - np.min(person_np, axis=0))
            if extent > max_extent:
                max_extent = extent
                main_idx = p_idx
        if debug_lines is not None:
            debug_lines.append(f"  >> Selected main_idx={main_idx} (3D_extent={max_extent:.1f})")
            debug_lines.append("")
        break

    tracked_nlf_indices = []
    prev_pelvis = None
    prev_tracked_idx = None

    if debug_lines is not None:
        debug_lines.append("--- Per-Frame Tracking ---")

    for frame_idx, frame_poses in enumerate(pose_input):
        if frame_poses.shape[0] == 0:
            tracked_nlf_indices.append(0)
            if debug_lines is not None:
                debug_lines.append(f"Frame {frame_idx}: 0 people, tracked_idx=0 (empty)")
            continue
        if prev_pelvis is None:
            tracked_idx = main_idx if main_idx < frame_poses.shape[0] else 0
            if debug_lines is not None:
                pelvis = frame_poses[tracked_idx][0]
                p_np = pelvis.cpu().numpy() if isinstance(pelvis, torch.Tensor) else pelvis
                debug_lines.append(
                    f"Frame {frame_idx}: {frame_poses.shape[0]} people, tracked_idx={tracked_idx} (initial=main_idx), "
                    f"pelvis=({p_np[0]:.1f}, {p_np[1]:.1f}, {p_np[2]:.1f})"
                )
        else:
            min_dist = float('inf')
            tracked_idx = 0
            all_dists = []
            for p_idx in range(frame_poses.shape[0]):
                pelvis = frame_poses[p_idx][0]
                pelvis_np = pelvis.cpu().numpy() if isinstance(pelvis, torch.Tensor) else pelvis
                dist = np.linalg.norm(pelvis_np - prev_pelvis)
                all_dists.append((p_idx, dist, pelvis_np.copy()))
                if dist < min_dist:
                    min_dist = dist
                    tracked_idx = p_idx
            if debug_lines is not None:
                changed = "CHANGED" if prev_tracked_idx is not None and tracked_idx != prev_tracked_idx else "same"
                p_np = all_dists[tracked_idx][2]
                debug_lines.append(
                    f"Frame {frame_idx}: {frame_poses.shape[0]} people, tracked_idx={tracked_idx} ({changed}), "
                    f"dist={min_dist:.1f}, pelvis=({p_np[0]:.1f}, {p_np[1]:.1f}, {p_np[2]:.1f})"
                )
                if frame_poses.shape[0] > 1 and frame_idx < 5:
                    for pi, di, pv in all_dists:
                        debug_lines.append(f"    candidate {pi}: dist={di:.1f}, pelvis=({pv[0]:.1f}, {pv[1]:.1f}, {pv[2]:.1f})")

        tracked_nlf_indices.append(tracked_idx)
        pelvis = frame_poses[tracked_idx][0]
        prev_pelvis = pelvis.cpu().numpy() if isinstance(pelvis, torch.Tensor) else pelvis
        prev_tracked_idx = tracked_idx

    if debug_lines is not None:
        debug_lines.append("")

    filtered_pose_input = []
    for frame_idx, frame_poses in enumerate(pose_input):
        t_idx = tracked_nlf_indices[frame_idx]
        if frame_poses.shape[0] > 0 and t_idx < frame_poses.shape[0]:
            filtered_pose_input.append(frame_poses[t_idx:t_idx+1])
        elif frame_poses.shape[0] > 0:
            filtered_pose_input.append(frame_poses[0:1])
        else:
            filtered_pose_input.append(frame_poses)

    if dw_pose_input is not None:
        if debug_lines is not None:
            debug_lines.append("--- DWPose Matching ---")

        for frame_idx, frame_dw in enumerate(dw_pose_input):
            num_dw_people = frame_dw['bodies']['candidate'].shape[0]
            if num_dw_people <= 1:
                if debug_lines is not None and frame_idx < 3:
                    debug_lines.append(f"Frame {frame_idx}: {num_dw_people} DW people, skipped (<=1)")
                continue

            nlf_frame = filtered_pose_input[frame_idx]
            best_dw_idx = 0

            if debug_lines is not None:
                debug_lines.append(f"Frame {frame_idx}: {num_dw_people} DW people")

            if nlf_frame.shape[0] > 0:
                neck = nlf_frame[0][15]
                neck_np = neck.cpu().numpy() if isinstance(neck, torch.Tensor) else neck
                if np.sum(np.abs(neck_np)) > 0.01 and neck_np[2] > 0.01:
                    u = (fx * neck_np[0] / neck_np[2] + cx) / width
                    v = (fy * neck_np[1] / neck_np[2] + cy) / height
                    neck_2d = np.array([u, v])

                    if debug_lines is not None:
                        debug_lines.append(f"  NLF neck 3D=({neck_np[0]:.1f}, {neck_np[1]:.1f}, {neck_np[2]:.1f}) -> 2D=({u:.4f}, {v:.4f})")

                    min_dist = float('inf')
                    for dw_p_idx in range(num_dw_people):
                        dw_body = frame_dw['bodies']['candidate'][dw_p_idx]
                        valid = np.any(dw_body != 0, axis=1)
                        dw_center = np.mean(dw_body[valid], axis=0) if np.any(valid) else np.mean(dw_body, axis=0)
                        dist = np.linalg.norm(neck_2d - dw_center)
                        if debug_lines is not None:
                            debug_lines.append(f"  DW person {dw_p_idx} center=({dw_center[0]:.4f}, {dw_center[1]:.4f}), dist={dist:.4f}")
                        if dist < min_dist:
                            min_dist = dist
                            best_dw_idx = dw_p_idx

                    if debug_lines is not None:
                        debug_lines.append(f"  >> Matched DW person {best_dw_idx}")
                elif debug_lines is not None:
                    debug_lines.append(f"  NLF neck invalid (near-zero or Z<=0), defaulting to DW person 0")

            p = best_dw_idx
            frame_dw['bodies']['candidate'] = frame_dw['bodies']['candidate'][p:p+1]
            frame_dw['bodies']['subset'] = frame_dw['bodies']['subset'][p:p+1]
            frame_dw['hands'] = frame_dw['hands'][2*p:2*p+2]
            frame_dw['faces'] = frame_dw['faces'][p:p+1]
            if 'body_score' in frame_dw:
                frame_dw['body_score'] = frame_dw['body_score'][p:p+1]
            if 'hand_score' in frame_dw:
                frame_dw['hand_score'] = frame_dw['hand_score'][2*p:2*p+2]
            if 'face_score' in frame_dw:
                frame_dw['face_score'] = frame_dw['face_score'][p:p+1]

        if debug_lines is not None:
            debug_lines.append("")

    return filtered_pose_input, dw_pose_input

class RenderNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "Optional DW pose model for 2D drawing"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Optional reference DW pose model for alignment"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu", "tooltip": "Taichi device to use for rendering"}),
                "scale_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to scale hand keypoints when aligning DW poses"}),
                "render_backend": (["taichi", "torch"], {"default": "taichi", "tooltip": "Rendering backend to use"}),
                "single_person": ("BOOLEAN", {"default": False, "tooltip": "When True, select the main character from NLF (largest 3D body in first frame) and filter both NLF and DWPose to that one person."}),
            }
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, nlf_poses, width, height, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hands=True, render_device="gpu", scale_hands=True, render_backend="taichi", single_person=False):

        from .NLFPoseExtract.nlf_render import render_nlf_as_images, render_multi_nlf_as_images, shift_dwpose_according_to_nlf, process_data_to_COCO_format, intrinsic_matrix_from_field_of_view
        from .NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down
        if render_backend == "taichi":
            try:
                import taichi as ti
                device_map = {
                    "cpu": ti.cpu,
                    "gpu": ti.gpu,
                    "opengl": ti.opengl,
                    "cuda": ti.cuda,
                    "vulkan": ti.vulkan,
                    "metal": ti.metal,
                }
                ti.init(arch=device_map.get(render_device.lower()))
            except:
                logging.warning("Taichi selected but not installed. Falling back to torch rendering.")
                render_backend = "torch"

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        ori_camera_pose = intrinsic_matrix_from_field_of_view([height, width])
        ori_focal = ori_camera_pose[0, 0]

        debug_lines = [] if single_person else None

        if single_person:
            pose_input, dw_pose_input = filter_to_single_person(pose_input, dw_pose_input, ori_camera_pose, height, width, debug_lines=debug_lines)

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_pose_input is not None else 0

        if debug_lines is not None:
            debug_lines.append("=== CAMERA ALIGNMENT ===")
            debug_lines.append(f"num_people after filter: {num_people}")
            debug_lines.append(f"ori_focal: {ori_focal:.2f}")
            debug_lines.append(f"ori_camera_pose:\n{ori_camera_pose}")
            debug_lines.append("")

        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy()
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            if debug_lines is not None:
                pelvis_3d = pose_3d_first_driving_frame[0]
                debug_lines.append(f"First driving 3D pose (24 joints):")
                debug_lines.append(f"  pelvis (joint 0): ({pelvis_3d[0]:.1f}, {pelvis_3d[1]:.1f}, {pelvis_3d[2]:.1f})")
                debug_lines.append(f"  neck (joint 1): ({pose_3d_first_driving_frame[1][0]:.1f}, {pose_3d_first_driving_frame[1][1]:.1f}, {pose_3d_first_driving_frame[1][2]:.1f})")
                debug_lines.append(f"  head (joint 15): ({pose_3d_first_driving_frame[15][0]:.1f}, {pose_3d_first_driving_frame[15][1]:.1f}, {pose_3d_first_driving_frame[15][2]:.1f})")
                debug_lines.append("")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * height

            if debug_lines is not None:
                debug_lines.append(f"ref_dw_pose 2D (pixel coords, first 14 joints):")
                for ji in range(min(14, len(poses_2d_ref))):
                    debug_lines.append(f"  joint {ji}: ({poses_2d_ref[ji][0]:.1f}, {poses_2d_ref[ji][1]:.1f})")
                debug_lines.append("")

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices # align body or only lower body

            if debug_lines is not None:
                debug_lines.append(f"valid_upper_indices: {valid_upper_indices}")
                debug_lines.append(f"valid_lower_indices: {valid_lower_indices}")
                debug_lines.append(f"valid_indices (used for alignment): {valid_indices}")
                solver_name = "solve_new_camera_params_down" if len(valid_lower_indices) >= 4 else "solve_new_camera_params_central"
                debug_lines.append(f"solver: {solver_name}")
                debug_lines.append("")

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if debug_lines is not None:
                debug_lines.append(f"Alignment input - 3D COCO joints (selected):")
                for i, vi in enumerate(valid_indices):
                    j3 = pose_3d_coco_first_driving_frame[i]
                    j2 = pose_2d_ref[i]
                    debug_lines.append(f"  idx {vi}: 3D=({j3[0]:.1f}, {j3[1]:.1f}, {j3[2]:.1f}) <-> 2D=({j2[0]:.1f}, {j2[1]:.1f})")
                debug_lines.append("")

            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)
            else:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)

            scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   # poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2

            if debug_lines is not None:
                debug_lines.append(f"--- Solver Output ---")
                debug_lines.append(f"scale_m: {scale_m}")
                debug_lines.append(f"scale_s: {scale_s}")
                debug_lines.append(f"scale_face: {scale_face}")
                debug_lines.append(f"new_camera_intrinsics:\n{new_camera_intrinsics}")
                debug_lines.append("")

            logging.info(f"Scale - m: {scale_m}, face: {scale_face}")
            shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, height, width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)

            intrinsic_matrix = new_camera_intrinsics
        else:
            intrinsic_matrix = ori_camera_pose
            if debug_lines is not None:
                debug_lines.append("Skipped camera alignment (no ref_dw_pose or num_people != 1)")
                debug_lines.append("")

        render_fn = "render_multi_nlf_as_images" if pose_input[0].shape[0] > 1 else "render_nlf_as_images"
        if debug_lines is not None:
            debug_lines.append("=== RENDERING ===")
            debug_lines.append(f"render function: {render_fn}")
            debug_lines.append(f"final intrinsic_matrix:\n{intrinsic_matrix}")
            debug_lines.append(f"total frames: {len(pose_input)}")
            debug_lines.append("")

        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)

        if debug_lines is not None:
            try:
                output_dir = folder_paths.get_output_directory()
                debug_path = os.path.join(output_dir, "debug_single_person.txt")
                with open(debug_path, "w") as f:
                    f.write("\n".join(debug_lines))
                logging.info(f"Debug log written to {debug_path}")
            except Exception as e:
                logging.error(f"Failed to write debug log: {e}")

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

        return (frames_tensor.cpu().float(), mask.cpu().float())

class SaveNLFPosesAs3D:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "filename_prefix": ("STRING", {"default": "nlf_pose_3d"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 300.0, "step": 0.1, "tooltip": "Frames per second for the output animation"}),
            "cylinder_radius": ("FLOAT", {"default": 21.5, "tooltip": "Radius of the cylinders representing bones"}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_3d"
    CATEGORY = "WanVideoWrapper"

    def save_3d(self, nlf_poses, filename_prefix, fps, cylinder_radius):
        from .NLFPoseExtract.nlf_render import get_cylinder_specs_list_from_poses
        from .render_3d.export_utils import save_cylinder_specs_as_glb_animation
        try:
            if isinstance(nlf_poses, dict):
                pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
            else:
                pose_input = nlf_poses

            cylinder_specs_list = get_cylinder_specs_list_from_poses(pose_input, include_missing=True)
            logging.info(f"Generated {len(cylinder_specs_list)} frames of cylinder specs")

            output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(output_dir, filename_prefix)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            filename = f"{filename_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
            filepath = os.path.join(full_output_folder, filename)

            logging.info(f"Saving as GLB animation to {full_output_folder}")
            logging.info(f"Starting GLB animation export. Frames: {len(cylinder_specs_list)}")
            save_cylinder_specs_as_glb_animation(cylinder_specs_list, filepath, fps=fps, radius=cylinder_radius)
            logging.info(f"Saved GLB: {filepath}")
        except Exception as e:
            logging.error(f"Error in SaveNLFPosesAs3D: {e}")
            raise e

        return (filepath,)

NODE_CLASS_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": PoseDetectionVitPoseToDWPose,
    "RenderNLFPoses": RenderNLFPoses,
    "ConvertOpenPoseKeypointsToDWPose": ConvertOpenPoseKeypointsToDWPose,
    "SaveNLFPosesAs3D": SaveNLFPosesAs3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": "Pose Detection VitPose to DWPose",
    "RenderNLFPoses": "Render NLF Poses",
    "ConvertOpenPoseKeypointsToDWPose": "Convert OpenPose Keypoints to DWPose",
    "SaveNLFPosesAs3D": "Save NLF Poses as 3D Animation",
}
