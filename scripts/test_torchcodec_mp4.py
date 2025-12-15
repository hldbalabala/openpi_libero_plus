#!/usr/bin/env python3

"""
Test script to check if torchcodec can correctly parse a specified MP4 file.
Supports various video codecs including AV1, H.264, H.265, etc.

Example:
    python scripts/test_torchcodec_mp4.py /home/ubuntu/Desktop/hld/openpi/episode_000000_h264.mp4 --decode-frames

    python scripts/test_torchcodec_mp4.py /home/ubuntu/.cache/huggingface/lerobot/libero_plus_lerobot/videos/chunk-000/observation.images.front/episode_000000.mp4 --decode-frames
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    from torchcodec.decoders import VideoDecoder
except ImportError as exc:
    raise SystemExit(
        "torchcodec is not installed. Install via `uv pip install torchcodec` and retry."
    ) from exc

try:
    import torch
except ImportError:
    torch = None


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test if torchcodec can correctly parse a specified MP4 file."
    )
    parser.add_argument(
        "mp4_path",
        type=Path,
        help="Path to the MP4 file to test.",
    )
    parser.add_argument(
        "--decode-frames",
        action="store_true",
        help="If set, also attempt to decode a few frames to verify full functionality.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="Number of frames to decode when --decode-frames is set (default: 3).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--check-codec",
        action="store_true",
        help="If set, also check the video codec format using ffprobe (requires ffprobe to be installed).",
    )

    return parser.parse_args(argv)


def get_video_codec_info(video_path: Path) -> dict | None:
    """
    Get video codec information using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with codec information, or None if ffprobe is not available.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if video_stream:
            return {
                "codec_name": video_stream.get("codec_name", "unknown"),
                "codec_long_name": video_stream.get("codec_long_name", "unknown"),
                "codec_type": video_stream.get("codec_type", "unknown"),
                "width": video_stream.get("width", "unknown"),
                "height": video_stream.get("height", "unknown"),
                "r_frame_rate": video_stream.get("r_frame_rate", "unknown"),
                "duration": video_stream.get("duration", "unknown"),
            }
        return None
    except FileNotFoundError:
        LOGGER.debug("ffprobe not found, skipping codec detection.")
        return None
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("ffprobe failed: %s", exc)
        return None
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse ffprobe output: %s", exc)
        return None


def test_mp4_parsing(
    mp4_path: Path,
    *,
    decode_frames: bool = False,
    num_frames: int = 3,
    check_codec: bool = False,
) -> bool:
    """
    Test if torchcodec can correctly parse the specified MP4 file.

    Args:
        mp4_path: Path to the MP4 file.
        decode_frames: If True, also attempt to decode frames.
        num_frames: Number of frames to decode if decode_frames is True.
        check_codec: If True, also check the video codec format using ffprobe.

    Returns:
        True if parsing is successful, False otherwise.
    """
    if not mp4_path.exists():
        LOGGER.error("File does not exist: %s", mp4_path)
        return False

    if not mp4_path.is_file():
        LOGGER.error("Path is not a file: %s", mp4_path)
        return False

    # Check codec information if requested
    if check_codec:
        LOGGER.info("Checking video codec information...")
        codec_info = get_video_codec_info(mp4_path)
        if codec_info:
            LOGGER.info("Video codec information:")
            LOGGER.info("  - Codec: %s (%s)", codec_info.get("codec_name", "unknown"), codec_info.get("codec_long_name", "unknown"))
            LOGGER.info("  - Resolution: %sx%s", codec_info.get("width", "unknown"), codec_info.get("height", "unknown"))
            LOGGER.info("  - Frame rate: %s", codec_info.get("r_frame_rate", "unknown"))
            codec_name = codec_info.get("codec_name", "").lower()
            if codec_name == "av01" or "av1" in codec_name:
                LOGGER.info("  - Note: This is an AV1-encoded video.")
        else:
            LOGGER.warning("Could not retrieve codec information (ffprobe may not be installed).")

    try:
        LOGGER.info("Attempting to create VideoDecoder for: %s", mp4_path)
        decoder = VideoDecoder(str(mp4_path))

        LOGGER.info("Successfully created VideoDecoder.")
        LOGGER.info("Fetching video metadata...")

        metadata = decoder.metadata
        LOGGER.info("Video metadata retrieved successfully:")
        
        # Handle metadata as either dict or object
        def get_attr(obj, attr, default="unknown"):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)
        
        duration = get_attr(metadata, "duration", "unknown")
        fps = get_attr(metadata, "fps", "unknown")
        width = get_attr(metadata, "width", "unknown")
        height = get_attr(metadata, "height", "unknown")
        total_frames_meta = get_attr(metadata, "num_frames", "unknown")
        
        LOGGER.info("  - Duration: %s seconds", duration)
        LOGGER.info("  - FPS: %s", fps)
        LOGGER.info("  - Resolution: %sx%s", width, height)
        LOGGER.info("  - Total frames: %s", total_frames_meta)

        if decode_frames:
            LOGGER.info("Attempting to decode %d frame(s)...", num_frames)
            frames_decoded = 0
            # Determine how many frames to decode
            if total_frames_meta == "unknown":
                frames_to_decode = num_frames  # Use the requested number
            else:
                try:
                    total_frames_int = int(total_frames_meta)
                    frames_to_decode = min(num_frames, total_frames_int)
                except (ValueError, TypeError):
                    frames_to_decode = num_frames  # Use the requested number
            
            for i in range(frames_to_decode):
                try:
                    frame = decoder.get_frame_at(i)
                    if frame is not None:
                        frames_decoded += 1
                        # Try to get shape information from frame
                        # Frame might be a tensor or have different attributes
                        try:
                            if torch is not None and isinstance(frame, torch.Tensor):
                                frame_shape = tuple(frame.shape)
                            elif hasattr(frame, 'shape'):
                                frame_shape = frame.shape
                            elif hasattr(frame, 'tensor'):
                                frame_shape = tuple(frame.tensor.shape)
                            else:
                                frame_shape = "unknown"
                            LOGGER.info("  - Frame %d decoded successfully (shape: %s)", i, frame_shape)
                        except Exception:
                            LOGGER.info("  - Frame %d decoded successfully (type: %s)", i, type(frame).__name__)
                    else:
                        LOGGER.warning("  - Frame %d returned None", i)
                except Exception as frame_exc:
                    LOGGER.error("  - Failed to decode frame %d: %s", i, frame_exc)
                    return False

            if frames_decoded == num_frames:
                LOGGER.info("Successfully decoded all %d requested frames.", num_frames)
            else:
                LOGGER.warning("Only decoded %d out of %d requested frames.", frames_decoded, num_frames)

        LOGGER.info("✓ MP4 file can be correctly parsed by torchcodec.")
        return True

    except Exception as exc:
        LOGGER.error("✗ Failed to parse MP4 file: %s", exc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return False


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    mp4_path = args.mp4_path.expanduser().resolve()

    success = test_mp4_parsing(
        mp4_path,
        decode_frames=args.decode_frames,
        num_frames=args.num_frames,
        check_codec=args.check_codec,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

