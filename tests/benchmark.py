import torch
from tabulate import tabulate

import neural_capability_maps.dataset.se3 as se3
from neural_capability_maps.dataset.capability_map import sample_poses_in_reach, estimate_capability_map
from neural_capability_maps.dataset.kinematics import analytical_inverse_kinematics
from neural_capability_maps.dataset.morphology import sample_morph

from neural_capability_maps.logger import binary_confusion_matrix

torch.manual_seed(1)

morphs = sample_morph(10, 6, True)

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
    print(f"Morph_IDX: {morph_idx}")
    cell_indices = se3.index(sample_poses_in_reach(100_000, morph))
    _, manipulability = analytical_inverse_kinematics(morph, se3.cell(cell_indices.to(morph.device)))
    ground_truth = manipulability != -1
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
        print(true_positives, false_negatives, false_positives, true_negatives)
        print(benchmarks[-1])
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
print(tabulate(list(zip(tp, tn, fp, fn, f1, acc, reachable, minutes)),
               headers=["True Positives", "True Negatives", "False Positives", "False Negatives",
                        "F1 Score", "Accuracy", "Reachable", "Minutes"], floatfmt=".2f", tablefmt="github"))
