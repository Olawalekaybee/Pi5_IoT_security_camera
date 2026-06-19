"""Unit tests for the Person Re-ID module — runs in CPU mock mode, no NPU required."""

import pytest
import tempfile
import shutil
import numpy as np
from src.reid.identifier import PersonIdentifier, cosine_similarity


@pytest.fixture
def identifier():
    tmp_dir = tempfile.mkdtemp()
    ident = PersonIdentifier(model_path="models/mock_reid.hef", known_faces_dir=tmp_dir, threshold=0.75)
    yield ident
    ident.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)


class TestPersonIdentifier:
    def test_unknown_when_gallery_empty(self, identifier):
        crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        person_id, confidence = identifier.identify(crop)
        assert person_id is None
        assert confidence == 0.0

    def test_enroll_and_identify_same_crop(self, identifier):
        crop = np.full((100, 50, 3), 128, dtype=np.uint8)
        identifier.enroll("alice", crop)
        person_id, confidence = identifier.identify(crop)
        assert person_id == "alice"
        assert confidence >= 0.75

    def test_gallery_persists_to_disk(self, identifier):
        crop = np.full((100, 50, 3), 200, dtype=np.uint8)
        identifier.enroll("bob", crop)
        assert (identifier.known_faces_dir / "embeddings.json").exists()

    def test_empty_crop_does_not_crash(self, identifier):
        empty_crop = np.empty((0, 0, 3), dtype=np.uint8)
        person_id, confidence = identifier.identify(empty_crop)
        assert confidence == 0.0
