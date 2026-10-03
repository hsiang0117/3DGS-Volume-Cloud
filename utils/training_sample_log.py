"""Record consumed training frames using true dataset metadata, with no GPU work."""
import hashlib
import json
import os
from pathlib import Path

SAMPLING_PROTOCOL = "camera_prefetch_retry_same_camera_v1"


class TrainingSampleRecorder:
    def __init__(self, output, source, expected_iterations, component_ablation, stage):
        self.output = Path(output)
        self.source = Path(source).resolve()
        self.expected = int(expected_iterations)
        self.mode, self.stage = component_ablation, stage
        self._by_path, self._metadata = {}, {}
        for split in ("train", "test"):
            path = self.source / f"transforms_{split}.json"
            if not path.exists():
                continue
            data = path.read_bytes()
            self._metadata[split] = hashlib.sha256(data).hexdigest()
            for frame in json.loads(data)["frames"]:
                image_path = self.source / frame["file_path"].replace("\\", "/")
                if not image_path.suffix:
                    image_path = Path(str(image_path) + ".png")
                image_path = image_path.resolve()
                relative = image_path.relative_to(self.source).as_posix()
                key = os.path.normcase(str(image_path))
                if key in self._by_path:
                    raise ValueError(f"Duplicate train/test source image: {relative}")
                self._by_path[key] = (relative, int(frame["camera_index"]), int(frame["time_index"]), split)
        if "train" not in self._metadata:
            raise ValueError("Sampling records require transforms_train.json")
        self.log_path = self.output / "training_sampling.jsonl"
        self.manifest_path = self.output / "training_sampling_manifest.json"
        self._file = self.log_path.open("x", encoding="utf-8", newline="\n")
        self._digest = hashlib.sha256()
        self.count, self.split_counts = 0, {"train": 0, "test": 0}
        self._write_manifest(completed=False)

    def record(self, iteration, camera):
        if int(iteration) != self.count + 1:
            raise ValueError(f"Nonconsecutive sampling iteration: {iteration}")
        key = os.path.normcase(str(Path(camera.image_path).resolve()))
        if key not in self._by_path:
            raise ValueError(f"Camera lacks canonical metadata: {camera.image_path}")
        relative, cam, time, split = self._by_path[key]
        row = dict(iteration=int(iteration), file_path=relative, cam_index=cam,
                   time_index=time, image_name=str(camera.image_name))
        line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._file.write(line)
        self._digest.update(line.encode("utf-8"))
        self.count += 1
        self.split_counts[split] += 1
        if self.count % 1000 == 0:
            self._file.flush()

    def _write_manifest(self, completed):
        record = dict(protocol=SAMPLING_PROTOCOL,
                      seed=0, seed_origin="train.py safe_state() seeds Python/NumPy/Torch to 0",
                      stage=self.stage, component_ablation=self.mode,
                      expected_iterations=self.expected, consumed_iterations=self.count,
                      completed=completed, source=str(self.source),
                      transforms_sha256_by_split=self._metadata,
                      consumed_source_split_counts=self.split_counts,
                      sequence_log=str(self.log_path), sequence_sha256=self._digest.hexdigest(),
                      hash_scope="Exact UTF-8 LF JSONL records: iteration, relative file_path, cam_index, time_index, image_name; no branch/run/absolute path")
        self.manifest_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def finish(self):
        self._file.flush()
        self._file.close()
        if self.count != self.expected:
            self._write_manifest(completed=False)
            raise ValueError(f"Consumed {self.count} frames, expected {self.expected}")
        if hashlib.sha256(self.log_path.read_bytes()).hexdigest() != self._digest.hexdigest():
            raise ValueError("Sampling JSONL bytes do not match the sequence SHA256")
        self._write_manifest(completed=True)
        print(f"[sampling] consumed={self.count} sha256={self._digest.hexdigest()} protocol={SAMPLING_PROTOCOL}", flush=True)
