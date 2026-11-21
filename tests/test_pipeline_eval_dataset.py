"""Offline tests for eval/pipeline/dataset.py (whole-pipeline evaluation, Step 1: dataset integrity).

Run from the project root:  python -B -m unittest tests.test_pipeline_eval_dataset -v
Standard library only. Every failure case is built in a temporary folder; the approved dataset is only READ.
No application import, no network, no model, no .env, no protected resume artifact.
"""
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_DIR = os.path.join(ROOT, "eval", "pipeline")
MODULE_PATH = os.path.join(PIPELINE_DIR, "dataset.py")

_spec = importlib.util.spec_from_file_location("pipeline_dataset", MODULE_PATH)
dataset = sys.modules.get("pipeline_dataset") or importlib.util.module_from_spec(_spec)
if "pipeline_dataset" not in sys.modules:
    sys.modules["pipeline_dataset"] = dataset
    _spec.loader.exec_module(dataset)

NULL_LABEL = ('{\n  "status": "FROZEN",\n  "scenario_id": "%s",\n'
              '  "frozen": {"sha256": null, "frozen_on": "2026-10-29", "approved_by": "tester"}\n}\n')


def digest_of(text):
    """The freeze convention, written out independently of the module under test."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def label_text(scenario_id, newline="\n"):
    """A frozen label whose stored digest is correct. newline='\\r\\n' writes the same file with Windows endings."""
    body = NULL_LABEL % scenario_id
    final = body.replace('"sha256": null', '"sha256": "%s"' % digest_of(body))
    return final.replace("\n", newline)


def scenario_text(scenario_id, content_hash="null"):
    return ('{\n  "status": "DRAFT — NOT FROZEN",\n  "scenario_id": "%s",\n  "content_hash": %s\n}\n'
            % (scenario_id, content_hash))


def write(directory, name, content):
    data = content.encode("utf-8") if isinstance(content, str) else content
    with open(os.path.join(directory, name), "wb") as handle:
        handle.write(data)


class TempDataset(unittest.TestCase):
    """Builds a valid dataset in a temporary folder; each test then breaks one thing."""

    IDS = ("alpha", "bravo", "charlie")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pipeline_dataset_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.scenarios = os.path.join(self.tmp, "scenarios")
        self.expected = os.path.join(self.tmp, "expected")
        os.makedirs(self.scenarios)
        os.makedirs(self.expected)
        for scenario_id in self.IDS:
            write(self.scenarios, scenario_id + ".json", scenario_text(scenario_id))
            write(self.expected, scenario_id + ".json", label_text(scenario_id))

    def load(self):
        return dataset.load_dataset(self.scenarios, self.expected)

    def problems(self):
        with self.assertRaises(dataset.DatasetError) as caught:
            self.load()
        return caught.exception.problems

    def assertProblem(self, fragment):
        problems = self.problems()
        self.assertTrue(any(fragment in p for p in problems), f"{fragment!r} not in {problems}")


class ValidFixture(TempDataset):
    def test_valid_dataset_loads(self):
        result = self.load()
        self.assertEqual(result["ids"], list(self.IDS))
        self.assertEqual(set(result["scenarios"]), set(self.IDS))
        self.assertEqual(set(result["labels"]), set(self.IDS))
        self.assertEqual(set(result["verified_hashes"]), set(self.IDS))

    def test_ordering_is_by_filename_regardless_of_creation_order(self):
        for scenario_id in ("zulu", "mike"):  # created after alpha..charlie, sort between and after them
            write(self.scenarios, scenario_id + ".json", scenario_text(scenario_id))
            write(self.expected, scenario_id + ".json", label_text(scenario_id))
        result = self.load()
        self.assertEqual(result["ids"], ["alpha", "bravo", "charlie", "mike", "zulu"])
        self.assertEqual(list(result["labels"]), result["ids"])
        self.assertEqual(list(result["scenarios"]), result["ids"])

    def test_scenarios_need_not_be_frozen(self):
        result = self.load()
        for data in result["scenarios"].values():
            self.assertIn("NOT FROZEN", data["status"])  # accepted as is

    def test_scenario_content_hash_may_be_a_wellformed_digest_but_is_not_verified(self):
        write(self.scenarios, "alpha.json", scenario_text("alpha", '"%s"' % ("a" * 64)))
        self.load()  # accepted: shape only


class StrictJson(TempDataset):
    def test_duplicate_key_is_rejected_in_a_scenario(self):
        write(self.scenarios, "alpha.json",
              '{"status": "x", "scenario_id": "alpha", "content_hash": null, "status": "y"}')
        self.assertProblem("scenarios/alpha.json: is not valid JSON: duplicate key 'status'")

    def test_nested_duplicate_key_is_rejected_in_a_label(self):
        text = label_text("alpha").replace('"approved_by": "tester"', '"approved_by": "a", "approved_by": "b"')
        write(self.expected, "alpha.json", text)
        self.assertProblem("expected/alpha.json: is not valid JSON: duplicate key 'approved_by'")

    def test_malformed_json_is_rejected(self):
        write(self.scenarios, "bravo.json", '{"scenario_id": "bravo", ')
        self.assertProblem("scenarios/bravo.json: is not valid JSON")

    def test_bom_is_rejected(self):
        write(self.expected, "charlie.json", b"\xef\xbb\xbf" + label_text("charlie").encode("utf-8"))
        self.assertProblem("expected/charlie.json: starts with a UTF-8 BOM")

    def test_invalid_utf8_is_rejected(self):
        write(self.scenarios, "alpha.json", b'{"scenario_id": "alpha", "content_hash": null, "x": "\xff"}')
        self.assertProblem("scenarios/alpha.json: is not valid UTF-8")

    def test_top_level_must_be_an_object(self):
        write(self.scenarios, "alpha.json", "[1, 2]")
        self.assertProblem("scenarios/alpha.json: top level is not a JSON object")


class IdsAndPairs(TempDataset):
    def test_scenario_id_must_match_filename(self):
        write(self.scenarios, "alpha.json", scenario_text("not_alpha"))
        self.assertProblem("scenarios/alpha.json: field 'scenario_id' is 'not_alpha' but the filename says 'alpha'")

    def test_label_id_must_match_filename(self):
        write(self.expected, "bravo.json", label_text("not_bravo"))
        self.assertProblem("expected/bravo.json: field 'scenario_id' is 'not_bravo' but the filename says 'bravo'")

    def test_missing_scenario_id_is_reported(self):
        write(self.scenarios, "alpha.json", '{"status": "x", "content_hash": null}')
        self.assertProblem("scenarios/alpha.json: field 'scenario_id' is missing")

    def test_missing_expected_file_is_reported(self):
        os.remove(os.path.join(self.expected, "bravo.json"))
        self.assertProblem("scenarios/bravo.json: no matching expected/bravo.json")

    def test_extra_expected_file_is_reported(self):
        write(self.expected, "delta.json", label_text("delta"))
        self.assertProblem("expected/delta.json: no matching scenarios/delta.json")

    def test_extra_scenario_file_is_reported(self):
        write(self.scenarios, "delta.json", scenario_text("delta"))
        self.assertProblem("scenarios/delta.json: no matching expected/delta.json")

    def test_duplicate_ids_are_reported(self):
        write(self.scenarios, "bravo.json", scenario_text("alpha"))  # bravo.json now claims id 'alpha'
        problems = self.problems()
        self.assertTrue(any("duplicates scenarios/alpha.json" in p for p in problems), problems)

    def test_non_json_files_are_ignored(self):
        write(self.scenarios, "notes.txt", "not a dataset file")
        write(self.expected, "README.md", "not a dataset file")
        self.assertEqual(self.load()["ids"], list(self.IDS))

    def test_missing_folder_is_reported(self):
        shutil.rmtree(self.expected)
        self.assertProblem("expected folder")

    def test_empty_folder_is_reported(self):
        for name in os.listdir(self.scenarios):
            os.remove(os.path.join(self.scenarios, name))
        self.assertProblem("contains no .json files")


class FreezeState(TempDataset):
    def test_label_status_must_be_frozen(self):
        write(self.expected, "alpha.json", label_text("alpha").replace('"FROZEN"', '"DRAFT — NOT FROZEN"'))
        self.assertProblem("expected/alpha.json: field 'status' is 'DRAFT — NOT FROZEN', expected 'FROZEN'")

    def test_missing_frozen_object_is_reported(self):
        write(self.expected, "alpha.json", '{"status": "FROZEN", "scenario_id": "alpha"}')
        self.assertProblem("expected/alpha.json: field 'frozen' is missing or not an object")

    def test_null_hash_is_rejected(self):
        write(self.expected, "alpha.json", NULL_LABEL % "alpha")
        self.assertProblem("expected/alpha.json: field 'frozen.sha256' is missing or not a 64-character")

    def test_malformed_hashes_are_rejected(self):
        good = label_text("alpha")
        stored = json.loads(good)["frozen"]["sha256"]
        for bad in ("abc", stored.upper(), stored + "0", "g" * 64):
            with self.subTest(bad=bad[:12]):
                write(self.expected, "alpha.json", good.replace(stored, bad))
                self.assertProblem("expected/alpha.json: field 'frozen.sha256' is missing or not a 64-character")

    def test_hash_mismatch_is_detected(self):
        good = label_text("alpha")
        write(self.expected, "alpha.json", good.replace('"approved_by": "tester"', '"approved_by": "someone else"'))
        problems = self.problems()
        self.assertTrue(any("expected/alpha.json: field 'frozen.sha256' is" in p and "hashes to" in p
                            for p in problems), problems)

    def test_hash_of_other_content_is_detected(self):
        stored = json.loads(label_text("bravo"))["frozen"]["sha256"]
        write(self.expected, "alpha.json", label_text("alpha").replace(
            json.loads(label_text("alpha"))["frozen"]["sha256"], stored))
        self.assertProblem("expected/alpha.json: field 'frozen.sha256' is " + stored)


class HashConvention(TempDataset):
    def test_crlf_file_verifies_under_lf_normalisation(self):
        write(self.expected, "alpha.json", label_text("alpha", newline="\r\n"))
        self.assertIn("alpha", self.load()["verified_hashes"])

    def test_crlf_and_lf_files_give_the_same_digest(self):
        lf = label_text("alpha").encode("utf-8")
        crlf = label_text("alpha", newline="\r\n").encode("utf-8")
        stored = json.loads(lf)["frozen"]["sha256"]
        self.assertEqual(dataset.frozen_digest(lf, stored), dataset.frozen_digest(crlf, stored))

    def test_digest_is_taken_with_the_stored_value_set_to_null(self):
        body = NULL_LABEL % "alpha"
        stored = digest_of(body)
        final = body.replace('"sha256": null', '"sha256": "%s"' % stored).encode("utf-8")
        self.assertEqual(dataset.frozen_digest(final, stored), stored)
        # hashing the file as stored (digest still in place) would give a different value
        self.assertNotEqual(hashlib.sha256(final).hexdigest(), stored)

    def test_stored_value_not_found_exactly_once_cannot_be_hashed(self):
        self.assertIsNone(dataset.frozen_digest(b'{"frozen": {"sha256": null}}', "a" * 64))
        twice = ('{"a": {"sha256": "%s"}, "b": {"sha256": "%s"}}' % ("a" * 64, "a" * 64)).encode("utf-8")
        self.assertIsNone(dataset.frozen_digest(twice, "a" * 64))

    def test_verified_digest_is_returned_by_load(self):
        stored = json.loads(label_text("alpha"))["frozen"]["sha256"]
        self.assertEqual(self.load()["verified_hashes"]["alpha"], stored)


class ScenarioHashShape(TempDataset):
    def test_missing_content_hash_is_reported(self):
        write(self.scenarios, "alpha.json", '{"status": "x", "scenario_id": "alpha"}')
        self.assertProblem("scenarios/alpha.json: field 'content_hash' is missing")

    def test_malformed_content_hash_is_reported(self):
        write(self.scenarios, "alpha.json", scenario_text("alpha", '"nothex"'))
        self.assertProblem("scenarios/alpha.json: field 'content_hash' must be null or a 64-character")


class ErrorReporting(TempDataset):
    def test_all_problems_are_reported_together(self):
        write(self.scenarios, "alpha.json", scenario_text("wrong"))
        os.remove(os.path.join(self.expected, "bravo.json"))
        write(self.expected, "charlie.json", NULL_LABEL % "charlie")
        problems = self.problems()
        self.assertGreaterEqual(len(problems), 3, problems)
        self.assertIn("3", str(dataset.DatasetError(["a", "b", "c"])))


class ApprovedDataset(unittest.TestCase):
    """The real dataset, read only."""

    def files(self):
        out = []
        for folder in ("scenarios", "expected"):
            directory = os.path.join(PIPELINE_DIR, folder)
            out.extend(os.path.join(directory, n) for n in sorted(os.listdir(directory)) if n.endswith(".json"))
        return out

    def fingerprint(self):
        out = {}
        for path in self.files():
            with open(path, "rb") as handle:
                out[path] = hashlib.sha256(handle.read()).hexdigest()
        return out

    def test_approved_dataset_validates_and_hashes_verify(self):
        before = self.fingerprint()
        result = dataset.load_dataset()
        self.assertEqual(len(result["ids"]), 10)
        self.assertEqual(len(result["verified_hashes"]), 10)
        for scenario_id, label in result["labels"].items():
            self.assertEqual(label["status"], "FROZEN")
            self.assertEqual(label["frozen"]["sha256"], result["verified_hashes"][scenario_id])
        for scenario in result["scenarios"].values():
            self.assertEqual(scenario["status"], "DRAFT — NOT FROZEN")
        self.assertEqual(result["ids"], sorted(result["ids"]))
        self.assertEqual(len(before), 20)
        self.assertEqual(self.fingerprint(), before, "validation must not change any dataset file")


class Portability(unittest.TestCase):
    def test_module_imports_only_the_standard_library_subset(self):
        with open(MODULE_PATH, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertEqual(names, {"hashlib", "json", "os", "re"})

    def test_module_does_not_open_files_for_writing(self):
        with open(MODULE_PATH, "r", encoding="utf-8") as handle:
            source = handle.read()
        for mode in ('"w"', '"wb"', '"a"', "'w'", "'wb'"):
            self.assertNotIn("open(path, " + mode, source)
        self.assertNotIn("os.remove", source)
        self.assertNotIn("shutil", source)


if __name__ == "__main__":
    unittest.main()
