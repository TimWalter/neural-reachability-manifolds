import torch
from tabulate import tabulate

import neural_capability_maps.dataset.se3 as se3
from neural_capability_maps.dataset.capability_map import estimate_reachable_ball, estimate_capability_map
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
minutes = []
reachable = []
benchmarks = []
for morph_idx, morph in enumerate(morphs):
    print(f"Morph_IDX: {morph_idx}")
    morph = morph.to("cuda")
    centre, radius = estimate_reachable_ball(morph)
    cell_indices = se3.index(se3.random_ball(100_000, centre, radius))
    _, manipulability = analytical_inverse_kinematics(morph, se3.cell(cell_indices.to(morph.device)))
    ground_truth = manipulability != -1
    cell_indices = cell_indices.cpu()

    minutes += [0]
    true_positives = 0.0
    r_indices = torch.empty(0, dtype=torch.int64)
    while true_positives < 95.0 and minutes[-1] < 10:
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
    reachable += [ground_truth.sum() / ground_truth.shape[0] * 100]

mean_benchmark = torch.stack(benchmarks).mean(dim=0, keepdim=True).tolist()
mean_benchmark[0][0] = int(mean_benchmark[0][0])
mean_benchmark[0][1] = int(mean_benchmark[0][1])
print(tabulate(mean_benchmark,
               headers=["Filled Cells", "Total Samples<br>(Speed)", "Efficiency<br>(Total)", "Efficiency<br>(Unique)",
                        "Efficiency<br>(Collision)"], floatfmt=".4f", intfmt=",", tablefmt="github"))
print(tabulate(list(zip(tp, tn, fp, fn, acc, reachable, minutes)),
               headers=["True Positives", "True Negatives", "False Positives", "False Negatives",
                        "Accuracy", "Reachable", "Minutes"], floatfmt=".2f", tablefmt="github"))
