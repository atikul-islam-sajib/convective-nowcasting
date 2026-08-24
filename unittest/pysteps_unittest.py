import unittest
import numpy as np
import pandas as pd

from pysteps_lk_extrapolation_fixed import (
    contingency_counts,
    csi_from_counts,
    ets_from_counts,
    rainrate_to_db,
    db_to_rainrate,
    compute_fss,
    compute_pooled_mean,
    Config,
)


class TestContingencyCounts(unittest.TestCase):
    def test_known_case(self):
        pred = np.array([20.0, 20.0, 0.0, 0.0])
        target = np.array([20.0, 0.0, 20.0, 0.0])
        hits, misses, fa, cn = contingency_counts(pred, target, threshold=10.0)
        self.assertEqual((hits, misses, fa, cn), (1.0, 1.0, 1.0, 1.0))

    def test_perfect_forecast(self):
        pred = np.array([20.0, 0.0, 30.0, 0.0])
        target = np.array([20.0, 0.0, 30.0, 0.0])
        hits, misses, fa, cn = contingency_counts(pred, target, threshold=10.0)
        self.assertEqual(misses, 0.0)
        self.assertEqual(fa, 0.0)


class TestCSIandETS(unittest.TestCase):
    def test_csi_perfect_forecast(self):
        self.assertAlmostEqual(csi_from_counts(hits=10, misses=0, false_alarms=0), 1.0)

    def test_csi_no_overlap(self):
        self.assertAlmostEqual(csi_from_counts(hits=0, misses=10, false_alarms=10), 0.0)

    def test_csi_no_events_is_nan(self):
        self.assertTrue(np.isnan(csi_from_counts(hits=0, misses=0, false_alarms=0)))

    def test_ets_perfect_forecast_less_than_or_equal_csi(self):
        hits, misses, fa, cn = 10, 5, 5, 80
        csi = csi_from_counts(hits, misses, fa)
        ets = ets_from_counts(hits, misses, fa, cn)
        self.assertLessEqual(ets, csi + 1e-9)

    def test_ets_no_total_is_nan(self):
        self.assertTrue(np.isnan(ets_from_counts(0, 0, 0, 0)))


class TestPooledAggregation(unittest.TestCase):
    def _build_metrics_df(self, cfg):
        row_a = {'model': 'PySteps_Radar', 'datetime': 'A', 'horizon_min': 15,
                  'MAE': 1.0, 'MSE': 1.0, 'RMSE': 1.0, 'PSNR': 30.0, 'SSIM': 0.9}
        row_b = {'model': 'PySteps_Radar', 'datetime': 'B', 'horizon_min': 15,
                  'MAE': 1.0, 'MSE': 1.0, 'RMSE': 1.0, 'PSNR': 30.0, 'SSIM': 0.9}
        for suffix in ['5', '15']:
            row_a[f'THR{suffix}_hits'] = 1
            row_a[f'THR{suffix}_misses'] = 0
            row_a[f'THR{suffix}_fa'] = 0
            row_a[f'THR{suffix}_cn'] = 0
            row_b[f'THR{suffix}_hits'] = 0
            row_b[f'THR{suffix}_misses'] = 99
            row_b[f'THR{suffix}_fa'] = 0
            row_b[f'THR{suffix}_cn'] = 0
        for w in cfg.fss_window_sizes:
            row_a[f'FSS{w}'] = 1.0
            row_b[f'FSS{w}'] = 0.0
        return pd.DataFrame([row_a, row_b])

    def test_pooled_not_naive_mean(self):
        cfg = Config()
        cfg.categorical_thresholds = [5.0, 15.0]
        metrics_df = self._build_metrics_df(cfg)
        mean_df = compute_pooled_mean(metrics_df, cfg)

        row = mean_df.iloc[0]
        naive_mean_would_be = 0.5
        correct_pooled = 1.0 / (1.0 + 99.0 + 0.0)  

        self.assertAlmostEqual(row['CSI@5'], correct_pooled, places=6)
        self.assertNotAlmostEqual(row['CSI@5'], naive_mean_would_be, places=2)

    def test_csi_mean_and_ets_mean_present(self):
        cfg = Config()
        cfg.categorical_thresholds = [5.0, 15.0]
        metrics_df = self._build_metrics_df(cfg)
        mean_df = compute_pooled_mean(metrics_df, cfg)
        self.assertIn('CSI_mean', mean_df.columns)
        self.assertIn('ETS_mean', mean_df.columns)
        row = mean_df.iloc[0]
        expected_mean = (row['CSI@5'] + row['CSI@15']) / 2.0
        self.assertAlmostEqual(row['CSI_mean'], expected_mean, places=6)


class TestDbRoundTrip(unittest.TestCase):
    def test_roundtrip_above_threshold(self):
        rain = np.array([0.0, 1.0, 5.0, 20.0, 50.0])
        cfg = Config()
        db = rainrate_to_db(rain, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
        recovered = db_to_rainrate(db, cfg)
        np.testing.assert_allclose(recovered[1:], rain[1:], rtol=1e-4)

    def test_below_threshold_becomes_zero(self):
        cfg = Config()
        rain = np.array([0.0, 0.01, 0.05]) 
        db = rainrate_to_db(rain, threshold=cfg.rain_threshold, zerovalue=cfg.zerovalue_db)
        recovered = db_to_rainrate(db, cfg)
        np.testing.assert_allclose(recovered, np.zeros_like(rain), atol=1e-6)


class TestFSS(unittest.TestCase):
    def test_identical_fields_give_fss_one(self):
        field = np.zeros((20, 20), dtype=np.float32)
        field[5:10, 5:10] = 20.0 
        mask = np.ones_like(field, dtype=bool)
        cfg = Config()
        scores = compute_fss(field, field.copy(), mask, threshold=10.0, window_sizes=(1, 5, 9))
        for w, val in scores.items():
            self.assertAlmostEqual(val, 1.0, places=6, msg=f'{w} should be 1.0 for identical fields')

    def test_completely_disjoint_fields_low_fss1(self):
        pred = np.zeros((20, 20), dtype=np.float32)
        target = np.zeros((20, 20), dtype=np.float32)
        pred[0:3, 0:3] = 20.0
        target[17:20, 17:20] = 20.0 
        mask = np.ones_like(pred, dtype=bool)
        cfg = Config()
        scores = compute_fss(pred, target, mask, threshold=10.0, window_sizes=(1,))
        self.assertLess(scores['FSS1'], 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)