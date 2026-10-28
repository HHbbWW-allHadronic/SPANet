from typing import Dict, Callable
import warnings
from collections import defaultdict

import numpy as np
import torch

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.jet_reconstruction.significance import (
    significance_metrics, find_kinematic_columns, pair_invariant_mass,
)


class JetReconstructionValidation(JetReconstructionNetwork):
    # ---- Weighted significance metric (S/sqrt(B) from detection scores), see significance.py ----
    # Event weights come from WEIGHTS/physical and must be at the FULL luminosity for the validation set. Our validate
    # file is a random 20% of the signal+ttbar file, so its weights sum to only 20% of the full-lumi yield.
    #   "auto" (default): per-class factor = (sum w over train + validation) / (sum w over validation), computed once
    #                     from the datasets. Puts each class's validation yield at its full-sample total, for ANY split
    #                     fraction. Assumes train + validation together are the whole sample: NOT valid if training
    #                     uses only part of its file (--limit_dataset) -- a warning is printed then.
    #   a number:         one constant for every event (e.g. 5.0 for a 20% split, 1.0 if the weights are already full-lumi).
    SIGNIFICANCE_WEIGHT_SCALE = "auto"
    SIGNIFICANCE_SIGNAL_LABEL = 1        # CLASSIFICATIONS/EVENT/process_label: 0 QCD, 1 signal, 2 ttbar
    SIGNIFICANCE_MIN_B_RAW = 5           # skip bins / cuts with fewer raw background events than this
    SIGNIFICANCE_PRIMARY = "signif/joint/binned_Z"   # also logged as `validation_significance` (monitor this one)
    # Method 3: invariant masses of the jet pairs SPANet itself assigned (m_H1, m_W1), used only for the mass-binned
    # variants signif/mass/*. Needs pt, eta, phi and e (or mass) among the Source features of the event yaml.
    SIGNIFICANCE_MASSES = True
    # Method 2 (equal signal-efficiency bins) and method 3 (PURE mass grid: 5 x 5 signal-quantile bins in (m_H1, m_W1),
    # no score involved), exposed as monitorable names. The mass x score cross (..._g5_e3) is still logged, not monitored.
    SIGNIFICANCE_MONITOR_ALIASES = {
        "validation_significance_eff": "signif/joint/binned_Z_e10",
        "validation_significance_mass": "signif/mass/binned_Z_g5",
    }

    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        if self.balance_particles:
            self.particle_index_tensor_np = self.particle_index_tensor.cpu().detach().numpy()
            self.particle_weights_tensor_np = self.particle_weights_tensor.cpu().detach().numpy()
        # self.validation_step_metrics_outputs = []
        self._sens_buffer = []
        self._sens_summary_printed = False
        self._sens_masses_printed = False
        self._sens_truth_printed = False
        self._sens_class_scales = None
        self._mass_setup = None          # resolved lazily: (source_index, columns) or False if unavailable

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            # "sensitivity": sk_metrics.recall_score,
            # "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            # "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights):
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        product_permutation_groups = [value.cpu().numpy() for value in self.product_permutation_tensors.values()]
        num_event_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.

        # Jet accuracies are kept in lists of arrays as opposed to a single NumPy array since each
        # resonance can have a different number of product permutations

        jet_accuracies = [np.zeros((len(product_permutation_groups[i]), num_event_permutations, batch_size), dtype=bool) for i in range(num_targets)]
        weighted_jet_accuracies = [np.zeros(jet_accuracy.shape, dtype=bool) for jet_accuracy in jet_accuracies]
        particle_accuracies = np.zeros((num_event_permutations, num_targets, batch_size), dtype=bool)
        for i, event_permutation in enumerate(event_permutation_group):
            for j, (prediction, target, weight) in enumerate(zip(jet_predictions, stacked_targets[event_permutation], stacked_weights[event_permutation])):
                for k, product_permutation in enumerate(product_permutation_groups[j]):
                    jet_accuracies[j][k, i] = np.all(prediction == target[..., product_permutation], axis=1)
                    weighted_jet_accuracies[j][k, i] = jet_accuracies[j][k, i] * weight

            particle_accuracies[i] = stacked_masks[event_permutation] == particle_predictions

        for i in range(num_targets):
            jet_accuracies[i] = jet_accuracies[i].max(axis=0)
            weighted_jet_accuracies[i] = weighted_jet_accuracies[i].max(axis=0)

        jet_accuracies = np.asarray(jet_accuracies).sum(axis=0)
        weighted_jet_accuracies = np.asarray(weighted_jet_accuracies).sum(axis=0)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        tot_target_weights = (stacked_masks * stacked_weights).sum(0)
        jet_accuracies = jet_accuracies.max(0)
        weighted_jet_accuracies = weighted_jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        has_targets = tot_target_weights > 0
        weighted_avg_jet_accuracy = weighted_jet_accuracies[has_targets] / tot_target_weights[has_targets]
        metrics["validation_average_jet_accuracy"] = np.mean(weighted_avg_jet_accuracy)

        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets = batch[:5]
        event_weights = getattr(batch, "event_weight", None)
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources)

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        stacked_weights = np.zeros((num_targets, batch_size), dtype=float)
        for i, (target, mask, weight) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()
            stacked_weights[i] = weight.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        self.buffer_significance_inputs(particle_scores, classification_targets, event_weights, batch_size,
                                        self._assigned_masses(sources, jet_predictions, batch_size),
                                        self._assigned_masses(sources, list(stacked_targets), batch_size, quiet=True))

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights))

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean(), sync_dist=True)

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean(), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step)

            absolute_deviation = delta
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean(), sync_dist=True)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value, sync_dist=True, on_epoch=True)

        # self.validation_step_metrics_outputs.append(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # ------------------------------------------------------------------------------------------------------------
    # Weighted significance: events are only buffered per step; S/sqrt(B) needs the WHOLE validation set, so it is
    # computed once in on_validation_epoch_end (self.log(on_epoch=True) would just average per-batch values).
    # Signal and background being separate events is irrelevant: the truth label splits them, order does not matter.
    # ------------------------------------------------------------------------------------------------------------
    def _detection_names(self, num_targets):
        try:
            names = list(self.training_dataset.event_info.product_particles.keys())
        except AttributeError:
            names = []
        return [names[i] if i < len(names) else f"t{i}" for i in range(num_targets)]

    def _resolve_mass_setup(self):
        """Find which source holds pt/eta/phi/e and at which feature positions. Cached; False = mass bins disabled."""
        if self._mass_setup is not None:
            return self._mass_setup
        setup = False
        try:
            info = self.training_dataset.event_info
            for si, name in enumerate(info.input_names):
                cols = find_kinematic_columns(info.input_features[name])
                if cols is not None:
                    setup = (si, cols)
                    break
            if setup is False:
                warnings.warn("Significance: no source has pt, eta, phi and e/mass -- mass-binned variants disabled")
        except Exception as e:  # never let the metric kill a training run
            warnings.warn(f"Significance: could not set up the mass variables ({e!r}) -- mass-binned variants disabled")
        self._mass_setup = setup
        return setup

    def _assigned_masses(self, sources, jet_predictions, batch_size, quiet=False):
        """(num_targets, batch) invariant masses of the jets SPANet assigned to each target (H1, W1, ...), or None."""
        if not self.SIGNIFICANCE_MASSES:
            return None
        setup = self._resolve_mass_setup()
        if not setup:
            return None
        si, cols = setup
        try:
            data, mask = sources[si][0], sources[si][1]
            data, mask = self._to_np(data), self._to_np(mask)
            return np.vstack([pair_invariant_mass(data, mask, np.asarray(pred).reshape(batch_size, -1), cols)
                              for pred in jet_predictions])
        except Exception as e:
            if quiet:                      # truth-mass diagnostic only: never disable the real mass variants
                return None
            if not getattr(self, "_mass_warned", False):
                self._mass_warned = True
                warnings.warn(f"Significance: assigned-mass computation failed ({e!r}) -- mass-binned variants disabled")
            self._mass_setup = False
            return None

    def buffer_significance_inputs(self, particle_scores, classification_targets, event_weights, batch_size, masses=None,
                                   truth_masses=None):
        label_key = next((k for k in classification_targets if k.endswith("process_label")), None)
        if label_key is None or event_weights is None:
            if not getattr(self, "_sens_warned", False):
                self._sens_warned = True
                warnings.warn(
                    "Significance metric DISABLED: "
                    + ("no 'process_label' in the classification targets (is it listed in the event yaml?)"
                       if label_key is None else "the batch has no event_weight (Batch/dataset not patched?)")
                    + " -- validation_significance will be logged as 0.0")
            return

        self._sens_buffer.append((
            np.asarray(particle_scores, dtype=np.float64).reshape(-1, batch_size),
            np.asarray(classification_targets[label_key]).reshape(-1),
            event_weights.detach().cpu().numpy().astype(np.float64).reshape(-1),
            None if masses is None else np.asarray(masses, dtype=np.float64).reshape(-1, batch_size),
            None if truth_masses is None else np.asarray(truth_masses, dtype=np.float64).reshape(-1, batch_size),
        ))

    @staticmethod
    def _to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def _auto_class_scales(self):
        """{label: (sum w train + sum w validation) / sum w validation}, from the two datasets (cached)."""
        if self._sens_class_scales is not None:
            return self._sens_class_scales

        scales = {}
        try:
            def label_and_weight(dataset):
                key = next(k for k in dataset.classifications if k.endswith("process_label"))
                return self._to_np(dataset.classifications[key]).reshape(-1), self._to_np(dataset.event_weights).astype(np.float64)

            lab_v, w_v = label_and_weight(self.validation_dataset)
            lab_t, w_t = label_and_weight(self.training_dataset)
            for c in np.unique(lab_v):
                sv, st = w_v[lab_v == c].sum(), w_t[lab_t == c].sum()
                scales[int(c)] = float((st + sv) / sv) if sv != 0 else 1.0
        except Exception as e:  # never let the metric kill a training run
            warnings.warn(f"Significance: could not compute the automatic weight scale ({e!r}); using 1.0 -- "
                          f"set SIGNIFICANCE_WEIGHT_SCALE to a number")
            scales = {}

        limit = getattr(getattr(self, "options", None), "dataset_limit", 1.0)
        if limit not in (None, 1.0, 1):
            warnings.warn(f"Significance: dataset_limit={limit} -- the automatic weight scale assumes train + validation "
                          f"are the whole sample, so yields will be too small. Set SIGNIFICANCE_WEIGHT_SCALE by hand.")
        self._sens_class_scales = scales
        return scales

    def _weight_scale_array(self, labels, testing=False):
        if testing:
            return np.ones(len(labels))                      # test dataset is not the complement of train
        if isinstance(self.SIGNIFICANCE_WEIGHT_SCALE, str):
            scales = self._auto_class_scales()
            return np.array([scales.get(int(c), 1.0) for c in labels])
        return np.full(len(labels), float(self.SIGNIFICANCE_WEIGHT_SCALE))

    def compute_significance(self, testing=False):
        if not self._sens_buffer:
            return {}

        scores = np.concatenate([b[0] for b in self._sens_buffer], axis=1)      # (num_targets, N)
        labels = np.concatenate([b[1] for b in self._sens_buffer])
        weights = np.concatenate([b[2] for b in self._sens_buffer]) * self._weight_scale_array(labels, testing)
        truth_parts = [b[4] for b in self._sens_buffer]
        truth_matrix = np.concatenate(truth_parts, axis=1) if all(m is not None for m in truth_parts) else None
        mass_parts = [b[3] for b in self._sens_buffer]
        mass_matrix = np.concatenate(mass_parts, axis=1) if all(m is not None for m in mass_parts) else None
        self._sens_buffer = []

        is_signal = labels == self.SIGNIFICANCE_SIGNAL_LABEL
        # Print once, at the first REAL validation epoch -- not in Lightning's 2-batch sanity check, which only sees a
        # few hundred events and would show meaningless partial sums.
        try:
            sanity = bool(self.trainer.sanity_checking)
        except Exception:
            sanity = False
        if not self._sens_summary_printed and not sanity:
            self._sens_summary_printed = True
            if isinstance(self.SIGNIFICANCE_WEIGHT_SCALE, str) and not testing:
                print("Significance weight scale (auto, per class): "
                      + ", ".join(f"class {c}: x{v:.4f}" for c, v in sorted(self._auto_class_scales().items())))
            print("Significance validation: " + ", ".join(
                f"class {c}: n={int((labels == c).sum())} sum(w)={weights[labels == c].sum():.4g}"
                for c in np.unique(labels)))
        if is_signal.sum() == 0 or (~is_signal).sum() == 0:
            return {}

        names = self._detection_names(scores.shape[0])
        score_sets = {name: scores[i] for i, name in enumerate(names)}
        score_sets["joint"] = np.prod(scores, axis=0) ** (1.0 / scores.shape[0])   # geometric mean of all detections, in [0, 1]

        masses = None
        if mass_matrix is not None and mass_matrix.shape[1] == len(labels):
            masses = {f"m_{n}": mass_matrix[i] for i, n in enumerate(names)}
            if not self._sens_masses_printed and not sanity:
                self._sens_masses_printed = True
                # Sanity check of the mass reconstruction: signal should peak near m_H ~ 125 and m_W ~ 80 GeV.
                print("Significance assigned masses, signal (weighted mean / valid fraction): " + ", ".join(
                    f"{k}: {np.average(v[is_signal & np.isfinite(v)], weights=weights[is_signal & np.isfinite(v)]):.1f} GeV"
                    f" / {np.isfinite(v[is_signal]).mean():.3f}" if (is_signal & np.isfinite(v)).any() else f"{k}: none"
                    for k, v in masses.items()))

        if masses is not None and truth_matrix is not None and not self._sens_truth_printed and not sanity:
            self._sens_truth_printed = True
            # Diagnostic: the same pair-mass built from the TRUTH assignment. If predicted ~ truth, the mass reconstruction
            # is right and the means just reflect the physics (e.g. an off-shell W*); if not, the assignment is still poor.
            parts = []
            for i, n in enumerate(names):
                ok = is_signal & np.isfinite(truth_matrix[i]) & np.isfinite(mass_matrix[i])
                if ok.any():
                    parts.append(f"m_{n}: predicted {np.average(mass_matrix[i][ok], weights=weights[ok]):.1f} vs truth "
                                 f"{np.average(truth_matrix[i][ok], weights=weights[ok]):.1f} GeV (signal, n={int(ok.sum())})")
            print("Significance mass diagnostic, " + "; ".join(parts))

        metrics = significance_metrics(score_sets, is_signal, weights, min_b_raw=self.SIGNIFICANCE_MIN_B_RAW, masses=masses)
        if self.SIGNIFICANCE_PRIMARY in metrics:
            metrics["validation_significance"] = metrics[self.SIGNIFICANCE_PRIMARY]
        return metrics

    def _log_significance(self, testing=False):
        metrics = self.compute_significance(testing)
        for name, value in metrics.items():
            if name != "validation_significance" and not np.isnan(value):
                self.log(name, float(value), sync_dist=False)

        # ModelCheckpoint(monitor="validation_significance") raises if the key is missing from the logged metrics, so
        # it is ALWAYS logged: 0.0 when it could not be computed (no signal / no background in the buffer, or no bin
        # with enough raw background events).
        primary = metrics.get("validation_significance", float("nan"))
        self.log("validation_significance", float(primary) if np.isfinite(primary) else 0.0, sync_dist=False)

        # Plain-name aliases for the other two binning methods so extra ModelCheckpoints can monitor them (a '/' in a
        # monitored key breaks the checkpoint filename template). Always logged, 0.0 when not computable.
        for alias, key in self.SIGNIFICANCE_MONITOR_ALIASES.items():
            v = metrics.get(key, float("nan"))
            self.log(alias, float(v) if np.isfinite(v) else 0.0, sync_dist=False)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._sens_buffer = []

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_significance()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._sens_buffer = []

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self._log_significance(testing=True)

#    def on_validation_epoch_end(self):
#        # merge metrics from different mini batches into one dict
#        metrics_merged = defaultdict(list) 
#        for m in self.validation_step_metrics_outputs:
#            for key, value in m.items():
#                metrics_merged[key].append(value)
#
#        # average each metric over number of mini batches
#        metrics_averaged = {}
#        for key, values in metrics_merged.items():
#            metrics_averaged[f"mean_{key}"] = np.mean(values)
#
#        # log metrics
#        for name, value in metrics_averaged.items():
#            if not np.isnan(value):
#                self.log(name, value, sync_dist=True)
#
#        self.validation_step_metrics_outputs.clear()