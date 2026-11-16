import json
import os
import struct
import tempfile
import unittest

from tools import rofl_decoded_check as chk
from tools import rofl_meta


def fake_detail():
    parts = [{"riotIdGameName": f"p{i}", "participantId": i, "championName": f"C{i}", "kills": i, "deaths": 1,
              "assists": 2, "totalMinionsKilled": 100, "neutralMinionsKilled": 5, "wardsPlaced": 9,
              "champLevel": 14, "goldEarned": 9000} for i in range(1, 11)]
    return {"info": {"participants": parts}}


def fake_timeline():
    frames = []
    for m in range(0, 4):
        pf = {str(i): {"position": {"x": 1000 * i + 100 * m, "y": 500}} for i in range(1, 11)}
        ev = [{"type": "WARD_PLACED", "creatorId": 1, "timestamp": m * 60000 + 30000, "wardType": "YELLOW_TRINKET"}] if m else []
        frames.append({"timestamp": m * 60000, "participantFrames": pf, "events": ev})
    return {"info": {"frames": frames}}


class RoflMetaTest(unittest.TestCase):
    def test_read_and_compare(self):
        stats = [{"RIOT_ID_GAME_NAME": f"p{i}", "CHAMPIONS_KILLED": str(i), "NUM_DEATHS": "1", "ASSISTS": "2",
                  "MINIONS_KILLED": "100", "NEUTRAL_MINIONS_KILLED": "5", "WARD_PLACED": "9", "LEVEL": "14",
                  "GOLD_EARNED": "9001"} for i in range(1, 11)]
        stats[0]["CHAMPIONS_KILLED"] = "7"
        meta = json.dumps({"gameLength": 1, "lastGameChunkId": 2, "lastKeyFrameId": 3,
                           "statsJson": json.dumps(stats)}).encode()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "KR-1.rofl")
            with open(path, "wb") as f:
                f.write(b"RIOT\x02\x00" + b"\x00" * 10 + b"16.21.829.8620\x01\x00" + b"\x28\xb5\x2f\xfd" * 8)
                f.write(meta + struct.pack("<I", len(meta)))
            self.assertEqual(rofl_meta.build(path), "16.21.829.8620")
            self.assertEqual(rofl_meta.match_id(path), "KR_1")
            m = rofl_meta.read_meta(path)
            self.assertEqual(len(m["stats"]), 10)
            ok, diffs = rofl_meta.compare(m, fake_detail())
        self.assertEqual(ok, 9)  # 골드 +1은 허용, p1 킬만 다름
        self.assertEqual(diffs, [("p1", "CHAMPIONS_KILLED", 7, 1)])


class DecodedCheckTest(unittest.TestCase):
    def test_positions_and_wards(self):
        raw = {"detail": fake_detail(), "timeline": fake_timeline()}
        pos = [{"match_id": "KR_1", "t_s": str(m * 60), "riot_id": "p2#KR1", "x": str(2000 + 100 * m + 50), "y": "500"}
               for m in range(4)]
        with tempfile.TemporaryDirectory() as d:
            chk.TRACKS = d
            with open(os.path.join(d, "KR_1_C2.csv"), "w") as f:
                f.write("time,x,y,valid\n" + "".join(f"{m * 60},{2000 + 100 * m},500,1\n" for m in range(4)))
            res = chk.check_positions("KR_1", pos, raw)["p2"]
        self.assertEqual(res["track"]["median"], 50)
        self.assertEqual(res["timeline"]["n"], 3)
        self.assertTrue(chk.passed(res["track"]))
        wards = [{"match_id": "KR_1", "t_place_s": "91", "t_end_s": "181", "x": "7000", "y": "7000",
                  "ward_type": "Yellow", "team": "100", "owner_riot_id": "p1"},
                 {"match_id": "KR_1", "t_place_s": "400", "t_end_s": "490", "x": "7000", "y": "99999",
                  "ward_type": "Control", "team": "100", "owner_riot_id": "p1"}]
        w = chk.check_wards("KR_1", wards, raw)
        self.assertEqual((w["decoded"], w["timeline"], w["matched"], w["kind_agree"], w["in_map"]), (2, 3, 1, 1.0, 1))


if __name__ == "__main__":
    unittest.main()
