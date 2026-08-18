import json
import tempfile
import unittest
from pathlib import Path

import pyomo.environ as pyo
import pandas as pd

from opt.operation import validate_sized_system
from opt.utils import load_sized_parameters, load_sized_parameters_from_artifact
from sizing import BESS, Load, MicrogridDesign, Parameters


class SizingCycleTests(unittest.TestCase):
    def test_resilience_constraints_enforce_outage_service_and_adequacy(self):
        config = {
            'costs': {},
            'EDS': {'Pmax': 2.0, 'Pmin': 1.0},
            'BESS': {
                'Emax_kwh': 10.0,
                'Pmax_kw': 5.0,
                'DoD_frac': 1.0,
                'eta_c': 1.0,
                'eta_d': 1.0,
                'crate_per_h': 0.5,
            },
            'PV': {},
            'Load': {'Pmax_kw': 2.0},
            'sizing': {
                'npv_years': 1,
                'minimum_outage_load_served_fraction': 0.5,
                'resilience_peak_load_kw': 2.0,
                'resilience_outage_energy_kwh': 2.0,
            },
        }
        params = Parameters(config)
        model = pyo.ConcreteModel()
        bundle = {
            'slots': [0, 1],
            'scenarios': ['s0'],
            'contingencies': ['c0', 'c_0'],
            'transitions': [(0, 1)],
            'windows': {'c0': [], 'c_0': [0, 1]},
            'before': {'c0': [], 'c_0': []},
            'dt_h': 0.5,
            'dt_h_map': {0: 0.5, 1: 0.5},
            'pi_s': {'s0': 1.0},
            'pi_c': {'c0': 0.9, 'c_0': 0.1},
            'f_load': {(0, 's0'): 1.0, (1, 's0'): 1.0},
        }
        params.build(model, bundle)
        BESS(config['BESS'], params).build(model)
        Load(config['Load'], params).build(model, bundle)

        model.P_L_shed[0, 's0', 'c_0', 1].value = 1.0
        service = model.OutageMinimumLoadService[0, 's0', 'c_0', 1]
        self.assertAlmostEqual(pyo.value(service.body - service.upper), 0.0)
        model.P_L_shed[0, 's0', 'c_0', 1].value = 1.1
        self.assertGreater(pyo.value(service.body - service.upper), 0.0)
        with self.assertRaises(KeyError):
            model.OutageMinimumLoadService[0, 's0', 'c0', 1]

        model.E_BESS_year[1].value = 2.0
        power = model.BESSResiliencePowerAdequacy[1]
        energy = model.BESSResilienceEnergyAdequacy[1]
        self.assertAlmostEqual(pyo.value(power.body - power.lower), 0.0)
        self.assertAlmostEqual(pyo.value(energy.body - energy.lower), 1.0)

    def test_invalid_resilience_service_fraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must be in'):
            Parameters({'sizing': {'minimum_outage_load_served_fraction': 1.01}})

    def test_last_period_action_closes_representative_day(self):
        config = {
            'costs': {},
            'EDS': {'Pmax': 2.0, 'Pmin': 1.0},
            'BESS': {
                'Emax_kwh': 10.0,
                'Pmax_kw': 5.0,
                'DoD_frac': 0.9,
                'eta_c': 1.0,
                'eta_d': 1.0,
                'E_init_kwh': 10.0,
            },
            'PV': {},
            'Load': {'Pmax_kw': 5.0},
            'sizing': {'npv_years': 1, 'E_BESS_size_max_kwh': 10.0},
        }
        params = Parameters(config)
        model = pyo.ConcreteModel()
        bundle = {
            'slots': [0, 1],
            'scenarios': ['s0'],
            'contingencies': ['c0'],
            'transitions': [(0, 1)],
            'windows': {'c0': []},
            'before': {'c0': []},
            'dt_h': 0.5,
            'dt_h_map': {0: 0.5, 1: 0.5},
            'pi_s': {'s0': 1.0},
            'pi_c': {'c0': 1.0},
        }
        params.build(model, bundle)
        BESS(config['BESS'], params).build(model)

        model.E_BESS_year[1].value = 10.0
        model.E_BESS_init[1].value = 5.0
        model.E_BESS[0, 's0', 'c0', 1].value = 5.0
        model.E_BESS[1, 's0', 'c0', 1].value = 4.0
        model.P_BESS_c[1, 's0', 'c0', 1].value = 2.0
        model.P_BESS_d[1, 's0', 'c0', 1].value = 0.0
        closure = model.BESSCycleClosure['s0', 'c0', 1]
        self.assertAlmostEqual(pyo.value(closure.body), 0.0)

        model.P_BESS_c[1, 's0', 'c0', 1].value = 0.0
        self.assertAlmostEqual(pyo.value(closure.body), -1.0)

        # A late outage can now use energy reserved earlier in the day:
        # the representative day starts at 5 kWh, reaches 10 kWh, and the
        # final-period discharge returns it to the common cyclic state.
        model.E_BESS[1, 's0', 'c0', 1].value = 10.0
        model.P_BESS_d[1, 's0', 'c0', 1].value = 10.0
        self.assertAlmostEqual(pyo.value(closure.body), 0.0)

    def test_small_sizing_model_solves_with_cycle(self):
        config = {
            'time': {},
            'costs': {
                'c_shed_per_kwh': 1000.0,
                'c_pv_curt_per_kwh': 0.0,
                'EDS': {'00:00': 0.4, '12:00': 0.4},
            },
            'EDS': {
                'Pmax': 10.0,
                'Pmin': 10.0,
                'outage_probability_pct': 0.0,
                'outage_duration_hours': 0.0,
            },
            'BESS': {
                'Emax_kwh': 10.0,
                'Pmax_kw': 5.0,
                'DoD_frac': 0.9,
                'eta_c': 0.95,
                'eta_d': 0.95,
                'E_init_kwh': 10.0,
                'capex_per_kwh': 1000.0,
            },
            'PV': {'capex_per_kw': 1000.0},
            'Load': {'Pmax_kw': 5.0},
            'sizing': {
                'npv_years': 1,
                'E_BESS_size_max_kwh': 10.0,
                'split': 'train',
            },
        }
        profiles = pd.DataFrame(
            {
                'group': ['all', 'all'],
                'split': ['train', 'train'],
                'cluster': [0, 0],
                'slot': [0, 1],
                'value': [0.5, 0.5],
            }
        )
        joint = pd.DataFrame(
            {
                'split': ['train'],
                'group_load': ['all'],
                'group_pv': ['all'],
                'cluster_load': [0],
                'cluster_pv': [0],
                'probability': [1.0],
            }
        )
        design = MicrogridDesign(
            config,
            df_pv=profiles.copy(),
            df_load=profiles.copy(),
            df_prob_joint=joint,
        )
        model = design.build()
        result = design.optimize(
            tee=False,
            time_limit=30,
            threads=1,
            solver_name='appsi_highs',
        )
        self.assertEqual(str(result.solver.termination_condition).lower(), 'optimal')
        for constraint in model.BESSCycleClosure.values():
            self.assertAlmostEqual(pyo.value(constraint.body), 0.0, places=7)
        sizing_results = design.get_results()
        audit = sizing_results['BESS_cycle_closure_max_abs_kwh']
        self.assertLessEqual(audit, 1e-7)
        if sizing_results['E_hat_BESS_kwh'] > 0.0:
            self.assertGreaterEqual(sizing_results['BESS_initial_soc_fraction'], 0.1)
            self.assertLessEqual(sizing_results['BESS_initial_soc_fraction'], 1.0)
        else:
            self.assertIsNone(sizing_results['BESS_initial_soc_fraction'])

    def test_zero_probability_scenarios_are_excluded_by_default(self):
        config = {
            'time': {},
            'costs': {'EDS': {'00:00': 0.4}},
            'EDS': {'Pmax': 2.0, 'Pmin': 1.0},
            'BESS': {'Emax_kwh': 2.0, 'Pmax_kw': 1.0, 'DoD_frac': 0.9},
            'PV': {},
            'Load': {'Pmax_kw': 1.0},
            'sizing': {'npv_years': 1},
        }
        profiles = pd.DataFrame(
            {
                'group': ['all', 'all'],
                'split': ['train', 'train'],
                'cluster': [0, 0],
                'slot': [0, 1],
                'value': [0.5, 0.5],
            }
        )
        joint = pd.DataFrame(
            {
                'split': ['train', 'train'],
                'group_load': ['all', 'all'],
                'group_pv': ['all', 'all'],
                'cluster_load': [0, 0],
                'cluster_pv': [0, 0],
                'probability': [1.0, 0.0],
            }
        )
        design = MicrogridDesign(
            config,
            df_pv=profiles.copy(),
            df_load=profiles.copy(),
            df_prob_joint=joint,
        )
        model = design.build()
        self.assertEqual(len(model.S), 1)
        self.assertTrue(design.get_results()['metadata']['positive_probability_scenarios_only'])


class SizedParameterTests(unittest.TestCase):
    def test_default_loader_uses_versioned_paper_artifact(self):
        params = load_sized_parameters()
        source = Path(params['sizing_case_applied']['source'])

        self.assertEqual(
            source.as_posix(),
            'paper/sizing/economic/degradation/sizing_decision_variables.json',
        )
        self.assertTrue(source.is_file())
        validate_sized_system(params)

    def test_portable_sizing_artifact_does_not_require_original_results_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / 'portable_sizing.json'
            artifact = {
                'case_id': 'critical_50',
                'has_loaded_solution': True,
                'metadata': {
                    'cyclic_daily_soc': True,
                    'optimized_cyclic_initial_soc': True,
                },
                'model_audit': {'cycle_closure_max_abs_kwh': 0.0},
                'decision_variables': {
                    'P_hat_PV_kw': 0.0,
                    'E_hat_BESS_kwh': 6.5,
                    'E_BESS_init_kwh': 3.25,
                },
            }
            artifact_path.write_text(json.dumps(artifact), encoding='utf-8')
            params = load_sized_parameters_from_artifact(
                'data/parameters.json', artifact_path
            )

            self.assertEqual(params['sizing_case_applied']['case'], 'critical_50')
            self.assertEqual(params['PV']['Pmax_kw'], 0.0)
            self.assertEqual(params['BESS']['Emax_kwh'], 6.5)
            self.assertAlmostEqual(params['BESS']['Pmax_kw'], 2.6)
            self.assertEqual(params['BESS']['E_init_kwh'], 3.25)
            validate_sized_system(params)

    def test_loader_rejects_pre_cyclic_sizing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / 'alpha_gt_0'
            case_dir.mkdir()
            artifact = {
                'has_loaded_solution': True,
                'decision_variables': {
                    'P_hat_PV_kw': 1.5,
                    'E_hat_BESS_kwh': 6.0,
                },
            }
            (case_dir / 'sizing_decision_variables.json').write_text(
                json.dumps(artifact), encoding='utf-8'
            )
            with self.assertRaisesRegex(ValueError, 'predates cyclic daily SoC closure'):
                load_sized_parameters(
                    params_json='data/parameters.json',
                    results_root=tmp,
                )

    def test_loader_applies_and_validates_sizing_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / 'alpha_gt_0'
            case_dir.mkdir()
            artifact = {
                'has_loaded_solution': True,
                'metadata': {
                    'cyclic_daily_soc': True,
                    'optimized_cyclic_initial_soc': True,
                },
                'model_audit': {'cycle_closure_max_abs_kwh': 0.0},
                'decision_variables': {
                    'P_hat_PV_kw': 1.5,
                    'E_hat_BESS_kwh': 6.0,
                    'E_BESS_init_kwh': 3.0,
                },
            }
            (case_dir / 'sizing_decision_variables.json').write_text(
                json.dumps(artifact), encoding='utf-8'
            )

            params = load_sized_parameters(
                params_json='data/parameters.json',
                results_root=str(root),
            )
            self.assertEqual(params['PV']['Pmax_kw'], 1.5)
            self.assertEqual(params['BESS']['Emax_kwh'], 6.0)
            self.assertAlmostEqual(params['BESS']['Pmax_kw'], 2.4)
            self.assertEqual(params['BESS']['E_init_kwh'], 3.0)
            validate_sized_system(params)

            params['PV']['Pmax_kw'] = 2.5
            with self.assertRaises(ValueError):
                validate_sized_system(params)


if __name__ == '__main__':
    unittest.main()
