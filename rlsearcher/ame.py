"""AME (Reinforcement Learning with HyperBand)"""
import pdb
import copy
import logging
import torch
import math
from typing import Dict, Optional

import ConfigSpace
from ray.tune.sample import Categorical, Domain, Float, Integer, LogUniform, \
    Normal, \
    Quantized, \
    Uniform
from ray.tune.suggest import Searcher
from ray.tune.suggest.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import unflatten_dict

logger = logging.getLogger(__name__)


class _AMEJobWrapper():
    """Mock object for HpBandSter to process."""

    def __init__(self, loss: float, budget: float, config: dict):
        self.result = {"loss": loss}
        self.kwargs = {"budget": budget, "config": config.copy()}
        self.exception = None


class TuneAME(Searcher):
    """AME suggestion component.


    Requires HpBandSter and ConfigSpace to be installed. You can install
    HpBandSter and ConfigSpace with: ``pip install hpbandster ConfigSpace``.

    This should be used in conjunction with HyperBandForAME.

    Args:
        space (ConfigurationSpace): Continuous ConfigSpace search space.
            Parameters will be sampled from this space which will be used
            to run trials.
        ame_config (dict): configuration for HpBandSter AME algorithm
        max_concurrent (int): Number of maximum concurrent trials. Defaults
            to 10.
        metric (str): The training result objective value attribute.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.

    Tune automatically converts search spaces to TuneAME's format:

    .. code-block:: python

        config = {
            "width": tune.uniform(0, 20),
            "height": tune.uniform(-100, 100),
            "activation": tune.choice(["relu", "tanh"])
        }

        algo = TuneAME(max_concurrent=4, metric="mean_loss", mode="min")
        ame = HyperBandForAME(
            time_attr="training_iteration",
            metric="mean_loss",
            mode="min",
            max_t=100)
        run(my_trainable, config=config, scheduler=ame, search_alg=algo)

    If you would like to pass the search space manually, the code would
    look like this:

    .. code-block:: python

        import ConfigSpace as CS

        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter("width", lower=0, upper=20))
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter("height", lower=-100, upper=100))
        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                name="activation", choices=["relu", "tanh"]))

        algo = TuneAME(
            config_space, max_concurrent=4, metric="mean_loss", mode="min")
        ame = HyperBandForAME(
            time_attr="training_iteration",
            metric="mean_loss",
            mode="min",
            max_t=100)
        run(my_trainable, scheduler=ame, search_alg=algo)

    """

    def __init__(self,
                 space: Optional[ConfigSpace.ConfigurationSpace] = None,
                 ame_config: Optional[Dict] = None,
                 max_concurrent: int = 10,
                 metric: Optional[str] = None,
                 mode: Optional[str] = None):
        from rlsearcher.agent_ame import AME
        assert AME is not None, "rlsearcher/agent_ame.py not found"
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
        self._max_concurrent = max_concurrent
        self.trial_to_params = {}
        self.running = set()
        self._metric = metric

        self._ame_config = ame_config
        self._space = space

        super(TuneAME, self).__init__(metric=self._metric, mode=mode)

        if self._space:
            self.setup_ame()

    def setup_ame(self):
        from rlsearcher.agent_ame import AME

        if self._mode == "max":
            self._metric_op = -1.
        elif self._mode == "min":
            self._metric_op = 1.

        ame_config = self._ame_config or {}
        self.amer = AME(self._space, **ame_config)

    def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                              config: Dict) -> bool:
        if self._space:
            return False
        space = self.convert_search_space(config)
        self._space = space

        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        self.setup_ame()
        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if not self._space:
            raise RuntimeError(
                "Trying to sample a configuration from {}, but no search "
                "space has been defined. Either pass the `{}` argument when "
                "instantiating the search algorithm, or pass a `config` to "
                "`tune.run()`.".format(self.__class__.__name__, "space"))

        if len(self.running) < self._max_concurrent:
            # This parameter is not used in hpbandster implementation.
            config, info = self.amer.get_config(None)
            self.trial_to_params[trial_id] = copy.deepcopy(config)
            self.running.add(trial_id)
            return unflatten_dict(config)
        return None

    def on_trial_result(self, trial_id: str, result: Dict):
        if result["training_iteration"] < 10:
            hbs_wrapper = self.to_wrapper(trial_id, result)
            self.amer.new_result(hbs_wrapper)

    def on_trial_complete(self,
                          trial_id: str,
                          result: Optional[Dict] = None,
                          error: bool = False):
        del self.trial_to_params[trial_id]
        if trial_id in self.running:
            self.running.remove(trial_id)

    def to_wrapper(self, trial_id: str, result: Dict) -> _AMEJobWrapper:
        return _AMEJobWrapper(self._metric_op * result[self.metric],
                               result["training_iteration"],
                               self.trial_to_params[trial_id])

    def save(self, checkpoint_path: str):
        torch.save(self.amer.actor_critic, checkpoint_path)

    @staticmethod
    def convert_search_space(spec: Dict) -> ConfigSpace.ConfigurationSpace:
        spec = flatten_dict(spec, prevent_delimiter=True)
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if grid_vars:
            raise ValueError(
                "Grid search parameters cannot be automatically converted "
                "to a TuneAME search space.")

        def resolve_value(par: str, domain: Domain
                          ) -> ConfigSpace.hyperparameters.Hyperparameter:
            quantize = None

            sampler = domain.get_sampler()
            if isinstance(sampler, Quantized):
                quantize = sampler.q
                sampler = sampler.sampler

            if isinstance(domain, Float):
                if isinstance(sampler, LogUniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    return ConfigSpace.UniformFloatHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=True)
                elif isinstance(sampler, Uniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    return ConfigSpace.UniformFloatHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=False)
                elif isinstance(sampler, Normal):
                    return ConfigSpace.NormalFloatHyperparameter(
                        par,
                        mu=sampler.mean,
                        sigma=sampler.sd,
                        q=quantize,
                        log=False)

            elif isinstance(domain, Integer):
                if isinstance(sampler, Uniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    return ConfigSpace.UniformIntegerHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=False)

            elif isinstance(domain, Categorical):
                if isinstance(sampler, Uniform):
                    return ConfigSpace.CategoricalHyperparameter(
                        par, choices=domain.categories)

            raise ValueError("TuneAME does not support parameters of type "
                             "`{}` with samplers of type `{}`".format(
                                 type(domain).__name__,
                                 type(domain.sampler).__name__))

        cs = ConfigSpace.ConfigurationSpace()
        for path, domain in domain_vars:
            par = "/".join(path)
            value = resolve_value(par, domain)
            cs.add_hyperparameter(value)

        return cs
