import unittest
import numpy as np
from joint_fsda_env import DA, FS, METHODS, JointFSDAEnv, make_demo_data


def act(env, mask=None, method=1, ratio=0.5, cluster=0):
    return {FS: np.ones(env.F, np.int8) if mask is None else mask,
            DA: {"method": method, "ratio": np.array([ratio], np.float32), "cluster": cluster}}


class TestEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_demo_data(n=1500, seed=3)

    def make(self, **kw):
        return JointFSDAEnv(*self.data, horizon=kw.pop("horizon", 5), **kw)

    # ---- interface contract
    def test_shapes_and_spaces(self):
        e = self.make()
        obs, _ = e.reset(seed=0)
        self.assertEqual(obs[FS].shape, (3 * e.F + 4,))
        self.assertEqual(obs[DA].shape, (3 * e.K + 4,))
        self.assertEqual(e.state().shape, (e.state_dim,))
        for a in e.possible_agents:
            self.assertTrue(e.observation_space(a).contains(obs[a]))
        self.assertTrue(e.state_space.contains(e.state()))
        for _ in range(5):
            self.assertTrue(e.action_space(FS).contains(e.action_space(FS).sample()))
            self.assertTrue(e.action_space(DA).contains(e.action_space(DA).sample()))

    def test_random_rollout_runs_to_horizon_and_obs_in_bounds(self):
        e = self.make(patience=99)
        e.reset(seed=1)
        steps, done = 0, False
        while e.agents:
            a = {x: e.action_space(x).sample() for x in e.agents}
            obs, r, term, trunc, info = e.step(a)
            steps += 1
            for x in obs:
                self.assertTrue(e.observation_space(x).contains(obs[x]), x)
            self.assertAlmostEqual(r[FS], r[DA])           # shared reward
            self.assertTrue(np.isfinite(r[FS]))
        self.assertEqual(steps, e.T)
        self.assertTrue(trunc[FS] and not term[FS])
        self.assertEqual(e.step({}), ({}, {}, {}, {}, {}))  # stepping a finished env is a no-op

    # ---- determinism and rollback
    def test_reset_is_deterministic_and_rolls_back(self):
        e = self.make()
        o1, _ = e.reset(seed=7)
        e.step(act(e, mask=np.r_[np.ones(5), np.zeros(e.F - 5)].astype(np.int8)))
        o2, _ = e.reset(seed=7)
        np.testing.assert_allclose(o1[FS], o2[FS])
        np.testing.assert_allclose(o1[DA], o2[DA])
        self.assertTrue(e.mask.all())
        self.assertEqual(e.n_synth, 0)
        self.assertEqual(e.t, 0)

    # ---- reward composition
    def test_reward_matches_formula(self):
        e = self.make(alpha=2.0, beta=0.7, gamma=0.3, delta=0.4)
        e.reset(seed=0)
        mask = (np.arange(e.F) < 10).astype(np.int8)
        _, r, _, _, info = e.step(act(e, mask=mask, method=2, ratio=0.6, cluster=1))
        i = info[FS]
        expect = 2.0 * i["d_f1"] + 0.7 * i["gmean"] - 0.3 * i["feat_frac"] - 0.4 * i["d_syn"]
        self.assertAlmostEqual(r[FS], expect, places=9)
        self.assertAlmostEqual(i["feat_frac"], 10 / e.F)
        self.assertGreater(i["d_syn"], 0.0)

    def test_no_augmentation_has_zero_distortion(self):
        e = self.make()
        e.reset(seed=0)
        _, _, _, _, info = e.step(act(e, method=0, ratio=1.0))
        self.assertEqual(info[DA]["d_syn"], 0.0)
        self.assertEqual(info[DA]["n_synth"], 0)

    def test_feature_penalty_monotone(self):
        e = self.make(alpha=0, beta=0, gamma=1.0, delta=0)
        e.reset(seed=0)
        _, r_full, *_ = e.step(act(e, method=0))
        e.reset(seed=0)
        _, r_half, *_ = e.step(act(e, mask=(np.arange(e.F) < e.F // 2).astype(np.int8), method=0))
        self.assertGreater(r_half[FS], r_full[FS])

    # ---- edge cases
    def test_empty_mask_is_repaired(self):
        e = self.make()
        e.reset(seed=0)
        _, r, _, _, info = e.step(act(e, mask=np.zeros(e.F, np.int8)))
        self.assertTrue(info[FS]["forced_feature"])
        self.assertEqual(info[FS]["n_features"], 1)
        self.assertTrue(np.isfinite(r[FS]))

    def test_ratio_clipped_and_all_methods_work(self):
        e = self.make(patience=99)
        e.reset(seed=0)
        for m in range(len(METHODS)):
            _, _, _, _, info = e.step(act(e, method=m, ratio=5.0, cluster=0))
            gap = int((e.y_train == 0).sum() - (e.y_train == 1).sum())
            self.assertLessEqual(info[FS]["n_synth"], gap)
            if not e.agents:
                e.reset(seed=0)

    def test_invalid_inputs_raise(self):
        e = self.make()
        e.reset(seed=0)
        with self.assertRaises(KeyError):
            e.step({FS: np.ones(e.F)})
        with self.assertRaises(ValueError):
            e.step(act(e, mask=np.ones(e.F + 1)))
        with self.assertRaises(ValueError):
            e.step(act(e, method=99))
        with self.assertRaises(ValueError):
            e.step(act(e, cluster=e.K))

    def test_tiny_minority_cluster_edge_case(self):
        X, y, Xv, yv = self.data
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0][:3]   # only 3 minority rows
        keep = np.concatenate([idx0, idx1])
        e = JointFSDAEnv(X[keep], y[keep], Xv, yv, horizon=3, n_subclusters=4)
        e.reset(seed=0)
        self.assertEqual(e.n_eff, 1)
        _, r, _, _, i = e.step(act(e, method=1, cluster=3))       # cluster beyond n_eff
        self.assertTrue(i[FS]["invalid_cluster"])
        _, r, _, _, i = e.step(act(e, method=1, cluster=0))       # smote falls back to noise
        self.assertTrue(np.isfinite(r[FS]))

    def test_patience_termination(self):
        e = self.make(patience=2, horizon=50)
        e.reset(seed=0)
        n = 0
        while e.agents:
            _, _, term, _, _ = e.step(act(e, method=0))   # identical action => no improvement
            n += 1
        self.assertTrue(term[FS])
        self.assertLessEqual(n, 3)

    def test_bad_data_rejected(self):
        X, y, Xv, yv = self.data
        Xbad = X.copy(); Xbad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            JointFSDAEnv(Xbad, y, Xv, yv)
        with self.assertRaises(ValueError):
            JointFSDAEnv(X, 1 - y, Xv, 1 - yv)            # label 1 is the majority


if __name__ == "__main__":
    unittest.main(verbosity=2)
