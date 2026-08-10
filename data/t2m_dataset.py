from os.path import join as pjoin
import json
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
from pathlib import Path


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def motion_edit_collate_fn(batch):
    batch.sort(key=lambda x: x[-1], reverse=True)
    return default_collate(batch)


def motion_edit_eval_collate_fn(batch):
    batch.sort(key=lambda x: x[6], reverse=True)
    return default_collate(batch)


def _resolve_motion_path(path_value, root_path: Path, manifest_dir: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        for base in (root_path, manifest_dir):
            candidate = (base / path).expanduser()
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"Motion file not found: {path_value}")


def _first_present(record, keys):
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _load_motion_edit_manifest(manifest_path: Path):
    text = manifest_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("samples", "data", "annotations", "records"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"JSON edit manifest must contain a list of samples: {manifest_path}")
        return payload

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            records.append(json.loads(line))
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(
                f"TSV edit manifest lines must be source<TAB>target<TAB>instruction, got: {line[:120]}"
            )
        records.append({"source": parts[0], "target": parts[1], "instruction": "\t".join(parts[2:])})
    return records


def _simple_motion_edit_tokens(text):
    words = str(text).strip().lower().replace(".", " ").replace(",", " ").split()
    return [f"{word}/OTHER" for word in words]


class MotionEditDataset(data.Dataset):
    """Source/target/instruction triplets for global motion editing training."""

    def __init__(self, opt, mean, std, manifest_file):
        self.opt = opt
        self.mean = mean
        self.std = std
        self.max_motion_length = int(opt.max_motion_length)
        self.unit_length = int(opt.unit_length)
        self.motion_normalized = bool(getattr(opt, "motion_normalized", False))
        self.min_motion_len = 40 if opt.dataset_name == "t2m" else 24

        root_path = Path(getattr(opt, "data_root", ".")).expanduser().resolve()
        manifest_path = Path(manifest_file).expanduser().resolve()
        manifest_dir = manifest_path.parent
        raw_records = _load_motion_edit_manifest(manifest_path)
        self.records = []
        skipped = 0
        for raw in tqdm(raw_records, desc="Loading edit manifest"):
            source_value = _first_present(
                raw,
                ("source_motion", "source_path", "source", "src_motion", "src_path", "src"),
            )
            target_value = _first_present(
                raw,
                ("target_motion", "target_path", "target", "tgt_motion", "tgt_path", "tgt"),
            )
            instruction = _first_present(raw, ("instruction", "edit_instruction", "edit", "text", "caption", "prompt"))
            if source_value is None or target_value is None or instruction is None:
                skipped += 1
                continue
            try:
                source_path = _resolve_motion_path(source_value, root_path, manifest_dir)
                target_path = _resolve_motion_path(target_value, root_path, manifest_dir)
                source_shape = np.load(source_path, mmap_mode="r").shape
                target_shape = np.load(target_path, mmap_mode="r").shape
            except Exception:
                skipped += 1
                continue
            if len(source_shape) != 2 or len(target_shape) != 2:
                skipped += 1
                continue
            usable_len = min(int(source_shape[0]), int(target_shape[0]), self.max_motion_length)
            if usable_len < self.min_motion_len:
                skipped += 1
                continue
            self.records.append({
                "source_path": source_path,
                "target_path": target_path,
                "instruction": str(instruction),
                "tokens": raw.get("tokens", None),
                "usable_len": usable_len,
            })
        if not self.records:
            raise RuntimeError(f"No valid edit samples found in {manifest_path}; skipped={skipped}")
        print(f"Total edit pairs {len(self.records)}, skipped {skipped}")

    def __len__(self):
        return len(self.records)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def _load_pair(self, record):
        source = np.load(record["source_path"]).astype(np.float32)
        target = np.load(record["target_path"]).astype(np.float32)
        if source.ndim != 2 or target.ndim != 2:
            raise RuntimeError("Motion edit arrays must be [T, D]")
        common_len = min(source.shape[0], target.shape[0], self.max_motion_length)
        unit_length = self.unit_length
        if unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (common_len // unit_length - 1) * unit_length
        else:
            m_length = (common_len // unit_length) * unit_length
        min_aligned_length = int(np.ceil(self.min_motion_len / float(unit_length)) * unit_length)
        max_aligned_length = (self.max_motion_length // unit_length) * unit_length
        m_length = min(m_length, max_aligned_length)
        m_length = max(m_length, min_aligned_length)
        if m_length > common_len:
            m_length = (common_len // unit_length) * unit_length
        if m_length < min_aligned_length:
            raise RuntimeError("Motion edit sample became too short after alignment")
        start = random.randint(0, common_len - m_length)
        source = source[start:start + m_length]
        target = target[start:start + m_length]
        if not self.motion_normalized:
            source = (source - self.mean) / self.std
            target = (target - self.mean) / self.std
        if m_length < self.max_motion_length:
            pad_len = self.max_motion_length - m_length
            source = np.concatenate([source, np.zeros((pad_len, source.shape[1]), dtype=source.dtype)], axis=0)
            target = np.concatenate([target, np.zeros((pad_len, target.shape[1]), dtype=target.dtype)], axis=0)
        return source, target, m_length

    def __getitem__(self, item):
        record = self.records[item]
        source, target, m_length = self._load_pair(record)
        return record["instruction"], source, target, m_length


class MotionEditDatasetEval(MotionEditDataset):
    """Evaluator-normalized source/target/instruction triplets for edit evaluation."""

    def __init__(self, opt, mean, std, manifest_file, w_vectorizer):
        super().__init__(opt, mean, std, manifest_file)
        self.w_vectorizer = w_vectorizer
        self.max_text_len = int(getattr(opt, "max_text_len", 20))

    def _vectorize_text(self, instruction, tokens):
        if isinstance(tokens, str):
            token_list = tokens.strip().split()
        elif isinstance(tokens, (list, tuple)):
            token_list = [str(token) for token in tokens]
        else:
            token_list = _simple_motion_edit_tokens(instruction)
        if len(token_list) < self.max_text_len:
            token_list = ["sos/OTHER"] + token_list + ["eos/OTHER"]
            sent_len = len(token_list)
            token_list = token_list + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            token_list = token_list[:self.max_text_len]
            token_list = ["sos/OTHER"] + token_list + ["eos/OTHER"]
            sent_len = len(token_list)
        pos_one_hots = []
        word_embeddings = []
        for token in token_list:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        return (
            np.concatenate(word_embeddings, axis=0),
            np.concatenate(pos_one_hots, axis=0),
            sent_len,
            "_".join(token_list),
        )

    def __getitem__(self, item):
        record = self.records[item]
        source, target, m_length = self._load_pair(record)
        word_embeddings, pos_one_hots, sent_len, token_string = self._vectorize_text(
            record["instruction"],
            record.get("tokens", None),
        )
        return (
            word_embeddings,
            pos_one_hots,
            record["instruction"],
            sent_len,
            source,
            target,
            m_length,
            token_string,
        )

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        unit_length = int(self.opt.unit_length)
        if unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // unit_length - 1) * unit_length
        elif coin2 == 'single':
            m_length = (m_length // unit_length) * unit_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        min_aligned_length = int(np.ceil(min_motion_len / float(unit_length)) * unit_length)
        max_aligned_length = (self.max_motion_length // unit_length) * unit_length
        max_available_length = (len(motion) // unit_length) * unit_length
        m_length = min(m_length, max_aligned_length, max_available_length)
        m_length = max(m_length, min_aligned_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        unit_length = int(self.opt.unit_length)
        if unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // unit_length - 1) * unit_length
        elif coin2 == 'single':
            m_length = (m_length // unit_length) * unit_length
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        min_aligned_length = int(np.ceil(min_motion_len / float(unit_length)) * unit_length)
        max_aligned_length = (self.max_motion_length // unit_length) * unit_length
        max_available_length = (len(motion) // unit_length) * unit_length
        m_length = min(m_length, max_aligned_length, max_available_length)
        m_length = max(m_length, min_aligned_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
