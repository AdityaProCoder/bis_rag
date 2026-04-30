import unittest

import inference


class TestInferenceSignals(unittest.TestCase):
    def test_part_parsing_numeric(self):
        s = inference.parse_query_signals("Need IS 10124 part 12 for fittings")
        self.assertEqual(s["part"], "12")

    def test_part_parsing_roman(self):
        self.assertEqual(inference.canonical_part("VII"), "7")

    def test_family_id_extract(self):
        self.assertEqual(inference._id_family("IS 10124 (PART 7)"), "10124")

    def test_near_id_penalty(self):
        q = inference.parse_query_signals("Need IS 736 for aluminium plate")
        cand = {
            "id": "737",
            "part": "",
            "title": "Wrought aluminium and aluminium alloy bars",
            "content": "",
        }
        cs = inference.candidate_signals(cand)
        score = inference.feature_score(q, cs)
        self.assertLess(score, 0)

    def test_product_mismatch_penalty(self):
        q = inference.parse_query_signals("Need aluminium plate standard")
        cand = {
            "id": "737",
            "part": "",
            "title": "Wrought aluminium and aluminium alloy bars, rods and sections",
            "content": "",
        }
        cs = inference.candidate_signals(cand)
        score = inference.feature_score(q, cs)
        self.assertLess(score, 0)


if __name__ == "__main__":
    unittest.main()
