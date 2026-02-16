import torch
from tabulate import tabulate

import numpy as np
from scipy.stats import bootstrap
def ci_95(data):
    if len(data) < 2:
        return (-1, -1)
    res = bootstrap((np.array(data),), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    return res.confidence_interval.low.astype(np.float64), res.confidence_interval.high.astype(np.float64)


import neural_capability_maps.dataset.r3 as r3
import neural_capability_maps.dataset.so3 as so3
import neural_capability_maps.dataset.se3 as se3

from neural_capability_maps.dataset.morphology import sample_morph
from neural_capability_maps.dataset.kinematics import analytical_inverse_kinematics, numerical_inverse_kinematics
from neural_capability_maps.dataset.capability_map import sample_poses_in_reach, estimate_capability_map
from neural_capability_maps.logger import binary_confusion_matrix

print(f"R3 Cells: {r3.N_CELLS} at {r3.DISTANCE_BETWEEN_CELLS}")
print(f"SO3 Cells: {so3.N_CELLS} at {so3.MIN_DISTANCE_BETWEEN_CELLS} - {so3.MAX_DISTANCE_BETWEEN_CELLS}")
print(f"SE3 Cells: {se3.N_CELLS} at {se3.MIN_DISTANCE_BETWEEN_CELLS} - {se3.MAX_DISTANCE_BETWEEN_CELLS}")

# Benchmark results on LRZ for 100 robots, analytically solvable and not
torch.manual_seed(1)
for analytically_solvable in [True, False]:
    morphs = sample_morph(100, 6, analytically_solvable)

    tp = []
    fn = []
    fp = []
    tn = []
    acc = []
    f1 = []
    minutes = []
    reachable = []
    benchmarks = []
    for morph_idx, morph in enumerate(morphs):
        cell_indices = se3.index(sample_poses_in_reach(100_000, morph))
        if analytically_solvable:
            _, manipulability = analytical_inverse_kinematics(morph, se3.cell(cell_indices.to(morph.device)))
        else:
            _, manipulability = numerical_inverse_kinematics(morph.to("cuda"), se3.cell(cell_indices.to(morph.device)).to("cuda"))
        ground_truth = manipulability.cpu() != -1
        reachable += [ground_truth.sum() / ground_truth.shape[0] * 100]

        cell_indices = cell_indices.cpu()

        morph = morph.to("cuda")
        minutes += [0]
        true_positives = 0.0
        r_indices = torch.empty(0, dtype=torch.int64)
        while true_positives < 95.0 and minutes[-1] < 30:
            new_r_indices, benchmark = estimate_capability_map(morph, True)
            r_indices = torch.cat([r_indices, new_r_indices]).unique()
            benchmarks += [torch.tensor(benchmark)]
            minutes[-1] += 1

            labels = torch.isin(cell_indices, r_indices)

            (true_positives, false_negatives), (false_positives, true_negatives) = binary_confusion_matrix(labels,
                                                                                                           ground_truth)
        tp += [true_positives]
        fn += [false_negatives]
        fp += [false_positives]
        tn += [true_negatives]
        acc += [(ground_truth == labels).sum() / labels.shape[0] * 100]
        f1 += [2 * true_positives / (2 * true_positives + false_positives + false_negatives) * 100]

    mean_benchmark = torch.stack(benchmarks).mean(dim=0, keepdim=True).tolist()
    mean_benchmark[0][0] = int(mean_benchmark[0][0])
    mean_benchmark[0][1] = int(mean_benchmark[0][1])
    print(tabulate(mean_benchmark,
                   headers=["Filled Cells", "Total Samples<br>(Speed)", "Efficiency<br>(Total)", "Efficiency<br>(Unique)",
                            "Efficiency<br>(Collision)"], floatfmt=".4f", intfmt=",", tablefmt="github"))
    mean_tp = sum(tp) / len(tp)
    mean_tn = sum(tn) / len(tn)
    mean_fp = sum(fp) / len(fp)
    mean_fn = sum(fn) / len(fn)
    mean_f1 = sum(f1) / len(f1)
    mean_acc = sum(acc) / len(acc)
    mean_reachable = sum(reachable) / len(reachable)
    mean_minutes = sum(minutes) / len(minutes)
    min_tp, max_tp = ci_95(tp)
    min_tn, max_tn = ci_95(tn)
    min_fp, max_fp = ci_95(fp)
    min_fn, max_fn = ci_95(fn)
    min_f1, max_f1 = ci_95(f1)
    min_acc, max_acc = ci_95(acc)
    min_reachable, max_reachable = ci_95(reachable)
    min_minutes, max_minutes = ci_95(minutes)
    print(tabulate([[mean_tp, mean_tn, mean_fp, mean_fn, mean_f1, mean_acc, mean_reachable, mean_minutes]],
                   headers=["True Positives", "True Negatives", "False Positives", "False Negatives",
                            "F1 Score", "Accuracy", "Reachable", "Minutes"], floatfmt=".2f", tablefmt="github"))
    print(tabulate([[min_tp, min_tn, min_fp, min_fn, min_f1, min_acc, min_reachable, min_minutes]],
                   headers=["True Positives", "True Negatives", "False Positives", "False Negatives",
                            "F1 Score", "Accuracy", "Reachable", "Minutes"], floatfmt=".2f", tablefmt="github"))
    print(tabulate([[max_tp, max_tn, max_fp, max_fn, max_f1, max_acc, max_reachable, max_minutes]],
                   headers=["True Positives", "True Negatives", "False Positives", "False Negatives",
                            "F1 Score", "Accuracy", "Reachable", "Minutes"], floatfmt=".2f", tablefmt="github"))
