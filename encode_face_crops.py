"""Encode isolated face crops and equal-scale controls on the archived OMG frames."""

import argparse
from pathlib import Path
import json
import pickle

import cv2
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from encode_omg import VISION, atomic_npz, frame_requests, matched_sections, read_requested_frames, sha256


DETECTOR_REVISION = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
DETECTOR_SHA256 = 'ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0'
DETECTOR_SIZE = 320
MIN_CONFIDENCE = .85
DOMINANT_AREA_RATIO = 1.5
CROP_SCALE = 1.6


def square_origin(x, y, size, width, height):
    return (int(np.clip(round(x - size / 2), 0, width - size)),
            int(np.clip(round(y - size / 2), 0, height - size)))


def crop_regions(image, detector):
    height, width = image.shape[:2]
    scale = DETECTOR_SIZE / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    canvas = np.zeros((DETECTOR_SIZE, DETECTOR_SIZE, 3), np.uint8)
    canvas[:h, :w] = resized
    _, boxes = detector.detect(canvas)
    if boxes is None:
        return None
    centers = boxes[:, :2] + boxes[:, 2:4] / 2
    boxes = boxes[((centers >= 0) & (centers < (w, h))).all(1) & (boxes[:, 2:4] > 0).all(1)]
    if not len(boxes):
        return None
    areas = boxes[:, 2] * boxes[:, 3]
    order = np.argsort(areas)
    if len(boxes) > 1 and areas[order[-1]] < DOMINANT_AREA_RATIO * areas[order[-2]]:
        return None

    x, y, box_width, box_height = boxes[order[-1], :4]
    size = min(height, width, max(1, round(CROP_SCALE * max(box_width, box_height) / scale)))
    left, top = square_origin((x + box_width / 2) * width / w, (y + box_height / 2) * height / h,
                              size, width, height)
    center_left, center_top = square_origin(width / 2, height / 2, size, width, height)
    face = image[top:top + size, left:left + size]
    control = image[center_top:center_top + size, center_left:center_left + size]
    intersection = max(size - abs(left - center_left), 0) * max(size - abs(top - center_top), 0)
    iou = intersection / (2 * size**2 - intersection)
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB), cv2.cvtColor(control, cv2.COLOR_BGR2RGB), iou


@torch.inference_mode()
def visual_vectors(images, model, processor, device):
    pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device=device, dtype=model.dtype)
    return model(pixel_values=pixels).pooler_output.cpu().numpy().astype(np.float16, copy=False)


def encode_section(section, media, whole, detector, model, processor, device, batch):
    video = cv2.VideoCapture(str(media / section['media']))
    detected = np.zeros(whole.shape[:2], bool)
    overlap = np.full(whole.shape[:2], np.nan, dtype=np.float32)
    face, center = whole.copy(), whole.copy()
    visible = np.isfinite(whole).all(-1) & np.any(whole != 0, axis=-1)
    seen = np.zeros_like(visible)
    check_images, references = [], []
    crops, destinations = [], []

    def flush():
        if crops:
            vectors = visual_vectors(crops, model, processor, device).reshape(-1, 2, whole.shape[-1])
            for (face_vector, control_vector), slots in zip(vectors, destinations, strict=True):
                for clip, slot in slots:
                    face[clip, slot] = face_vector
                    center[clip, slot] = control_vector
            crops.clear()
            destinations.clear()

    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or length < 1:
            raise ValueError(f'Cannot read matched media {section["media"]}.')
        requests = frame_requests(section['clips'], section['start'], fps, length)
        requests = {frame: [(row, slot) for row, slot in slots if visible[row, slot]]
                    for frame, slots in requests.items()}
        requests = {frame: slots for frame, slots in requests.items() if slots}
        ordered = sorted(requests)
        replay_frames = {ordered[0], ordered[len(ordered) // 2], ordered[-1]}
        for frame, image in read_requested_frames(video, requests, fps):
            slots = requests[frame]
            for row, slot in slots:
                seen[row, slot] = True
            if frame in replay_frames:
                check_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                row, slot = slots[0]
                references.append(whole[row, slot])
            crop = crop_regions(image, detector)
            if crop is None:
                continue
            localized, control, iou = crop
            crops.extend([localized, control])
            destinations.append(slots)
            for row, slot in slots:
                detected[row, slot] = True
                overlap[row, slot] = iou
            if len(crops) >= batch:
                flush()
        flush()
    finally:
        video.release()

    if not np.array_equal(seen, visible):
        raise ValueError(f'Frame availability changed in {section["media"]}.')
    actual = visual_vectors(check_images, model, processor, device).astype(float)
    expected = np.asarray(references, dtype=float)
    cosine = (actual * expected).sum(1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
    if not np.all(cosine >= .9995):
        raise ValueError(f'Whole-frame replay failed in {section["media"]}: cosine {cosine.min():.6f}.')
    for vectors in (face, center):
        np.testing.assert_array_equal(visible, np.isfinite(vectors).all(-1) & np.any(vectors != 0, axis=-1))
    return dict(face=face, center=center, localized=detected, overlap=overlap, replay_cosine=cosine)


def write_representations(directory, raw, raw_rows, face, center):
    variants = ('face', 'center', 'face_scene', 'center_scene')
    for name in variants:
        cropped = center if name.startswith('center') else face
        keep_scene = name.endswith('_scene')
        packets = {}
        for role, packet in raw.items():
            region = cropped[raw_rows[role]]
            vision = ((packet['vision'].astype(np.float32) + region.astype(np.float32)) * .5).astype(np.float16) \
                if keep_scene else region
            packets[role] = dict(packet, vision=vision)
        path = directory / f'omg_{name}.pkl'
        temporary = path.with_suffix('.pkl.part')
        with temporary.open('wb') as stream:
            pickle.dump(packets, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


def run(args):
    if sha256(args.detector) != DETECTOR_SHA256:
        raise ValueError('Use the pinned YuNet detector weights.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / 'by_section'
    cache.mkdir(exist_ok=True)
    settings = dict(detector_revision=DETECTOR_REVISION, detector_sha256=DETECTOR_SHA256,
                    visual_encoder=VISION, detector_size=DETECTOR_SIZE, minimum_confidence=MIN_CONFIDENCE,
                    dominant_area_ratio=DOMINANT_AREA_RATIO, crop_scale=CROP_SCALE,
                    whole_frame_sha256=sha256(args.original), opencv=cv2.__version__)
    settings_path = args.output_dir / 'crop_settings.json'
    expected_settings = json.dumps(settings, indent=2)
    if settings_path.exists() and settings_path.read_text(encoding='utf-8') != expected_settings:
        raise ValueError('Use a fresh directory for different crop settings or original data.')
    settings_path.write_text(expected_settings, encoding='utf-8')

    with args.original.open('rb') as stream:
        raw = pickle.load(stream)
    names = np.concatenate([packet['id'] for packet in raw.values()])
    whole = np.concatenate([packet['vision'] for packet in raw.values()])
    manifest = json.loads((args.prepared / 'manifest.json').read_text(encoding='utf-8'))
    sections = matched_sections(manifest, names)
    rows = {clip: index for index, clip in enumerate(names)}

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    detector = cv2.FaceDetectorYN.create(str(args.detector), '', (DETECTOR_SIZE, DETECTOR_SIZE),
                                        MIN_CONFIDENCE, .3, 5000)
    pretrained = dict(revision=VISION[1], cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    processor = CLIPImageProcessor.from_pretrained(VISION[0], **pretrained)
    model = CLIPVisionModel.from_pretrained(VISION[0], **pretrained, weights_only=True, dtype=dtype).to(device).eval()
    face, center = whole.copy(), whole.copy()
    localized = np.zeros(whole.shape[:2], bool)
    overlap = np.full(whole.shape[:2], np.nan, np.float32)
    cosine = []
    for index, (section, indices) in enumerate(sections[:args.limit], 1):
        record = cache / Path(section['media']).with_suffix('.npz').name
        if not record.is_file():
            current = encode_section(section, args.prepared / 'media', whole[indices], detector, model,
                                     processor, device, args.batch)
            atomic_npz(record, ids=names[indices], **current)
        with np.load(record, allow_pickle=False) as packet:
            np.testing.assert_array_equal(packet['ids'], names[indices])
            face[indices], center[indices] = packet['face'], packet['center']
            localized[indices], overlap[indices] = packet['localized'], packet['overlap']
            cosine.extend(packet['replay_cosine'].tolist())
        if index == 1 or index % 20 == 0:
            print(f'Encoded or loaded {index}/{len(sections[:args.limit])} matched sections.', flush=True)
    if args.limit is not None:
        print(f'Smoke replay cosine >= {min(cosine):.7f}.', flush=True)
        return

    native = np.any(whole != 0, axis=-1)
    assert not localized[~native].any()
    raw_rows = {name: np.array([rows[clip] for clip in packet['id']]) for name, packet in raw.items()}
    report = dict(settings, clips=len(names), observed_frames=int(native.sum()),
                  localized_frames=int(localized.sum()), median_crop_iou=float(np.median(overlap[localized])),
                  min_replay_cosine=min(cosine), replay_frames=len(cosine),
                  splits={role: dict(clips=len(indices), observed_frames=int(native[indices].sum()),
                                     localized_frames=int(localized[indices].sum()),
                                     median_crop_iou=float(np.median(overlap[indices][localized[indices]])))
                          for role, indices in raw_rows.items()})
    write_representations(args.output_dir, raw, raw_rows, face, center)
    atomic_npz(args.output_dir / 'crop_coverage.npz', ids=names, native=native, localized=localized, overlap=overlap)
    summary = args.output_dir / 'crop_summary.json'
    temporary = summary.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(summary)
    print(f'{report["localized_frames"]:,}/{report["observed_frames"]:,} bins have face crops, '
          f'median crop IoU={report["median_crop_iou"]:.3f}.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--limit', type=int, help='Run a smoke trial on this many media sections.')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.batch < 2 or (args.limit is not None and args.limit < 1):
        parser.error('Use a batch size of at least two and a positive section limit.')
    run(args)


if __name__ == '__main__':
    main()
