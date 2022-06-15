import numpy as np
import xgboost
from sklearn.ensemble import RandomForestClassifier

from satreeverify.rf_utils import get_ens_thresh as rf_get_ens_thresh
from satreeverify.rf_utils import get_output as rf_get_output
from satreeverify.rf_utils import hard_attack as rf_hard_attack
from satreeverify.rf_utils import soft_attack as rf_soft_attack
from satreeverify.utils import create_var_x, get_x_adv
from satreeverify.xgb_utils import get_ens_thresh as xgb_get_ens_thresh
from satreeverify.xgb_utils import get_output as xgb_get_output
from satreeverify.xgb_utils import hard_attack as xgb_hard_attack
from satreeverify.xgb_utils import soft_attack as xgb_soft_attack


class SATreeAttack:
    def __init__(self, model):
        self.model = model
        if isinstance(self.model, xgboost.sklearn.XGBClassifier):
            dump = model.get_booster().get_dump(dump_format="json")
            self.all_thresh = xgb_get_ens_thresh(dump)
        elif isinstance(self.model, RandomForestClassifier):
            self.all_thresh = rf_get_ens_thresh(model)
        else:
            raise NotImplementedError(f"Not implement for {type(model)}.")

        self._create_var_x()

    def _create_var_x(self):
        self.var_x = create_var_x(self.all_thresh)

    def soft_attack(self, sample, epsilon):
        if isinstance(self.model, xgboost.sklearn.XGBClassifier):
            s, c_weights = xgb_soft_attack(
                self.model, sample, epsilon, self.var_x
            )
            adv_weights = xgb_get_output(s, c_weights)
            pred = 1 / (
                1 + np.exp(-np.sum([v for k, v in adv_weights.items()]))
            )
        elif isinstance(self.model, RandomForestClassifier):
            s, c_weights = rf_soft_attack(
                self.model, sample, epsilon, self.var_x
            )
            adv_weights = rf_get_output(s, c_weights)
            pred = np.mean([v for k, v in adv_weights.items()])

        x_adv, x_adv_sample, compare = get_x_adv(s, self.var_x, sample)

        return {
            "adv_sample": x_adv_sample,
            "predi": pred,
            "comparison": compare,
            "sat_adv_sample": x_adv,
        }

    def hard_attack(self, sample, epsilon, nbits):
        if isinstance(self.model, xgboost.sklearn.XGBClassifier):
            s, c_weights, seq_nump, seq_numn = xgb_hard_attack(
                self.model, sample, epsilon, self.var_x, nbits
            )
            adv_weights = xgb_get_output(s, c_weights)
            pred = 1 / (
                1 + np.exp(-np.sum([v for k, v in adv_weights.items()]))
            )
        elif isinstance(self.model, RandomForestClassifier):
            s, c_weights, seq_num = rf_hard_attack(
                self.model, sample, epsilon, self.var_x, nbits
            )
            adv_weights = rf_get_output(s, c_weights)
            pred = np.mean([v for k, v in adv_weights.items()])

        x_adv, x_adv_sample, compare = get_x_adv(s, self.var_x, sample)

        return {
            "adv_sample": x_adv_sample,
            "predi": pred,
            "comparison": compare,
            "sat_adv_sample": x_adv,
        }
