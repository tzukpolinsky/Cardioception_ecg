import random
from itertools import product
import numpy as np

durations = [25, 35, 45, 100]
bpms = [50, 60, 70]


# Generate all 12 unique (duration, bpm) pairs
all_combinations = list(product(durations, bpms))

def create_blocks():
    for attempt in range(1000):
        remaining = all_combinations.copy()
        used_pairs = set()
        blocks = []

        for _ in range(3):  # 3 blocks
            block = []
            block_durations = set()
            block_bpms = []

            random.shuffle(remaining)

            for pair in remaining:
                dur, bpm = pair
                bpm_count = block_bpms.count(bpm)

                # Skip if already used, duration repeated, or bpm overused
                if pair in used_pairs or dur in block_durations or bpm_count >= 2:
                    continue

                # Add if fits constraints
                block.append(pair)
                block_durations.add(dur)
                block_bpms.append(bpm)
                used_pairs.add(pair)

                if len(block) == 4:
                    # Check BPM distribution: one repeated BPM, others once
                    bpm_counts = [block_bpms.count(b) for b in set(block_bpms)]
                    if sorted(bpm_counts) == [1, 1, 2]:
                        break
                    else:
                        # Invalid BPM pattern
                        for p in block:
                            used_pairs.discard(p)
                        block.clear()
                        block_durations.clear()
                        block_bpms.clear()

            if len(block) != 4:
                break  # Failed block
            blocks.append(block)

        if len(blocks) == 3:
            return blocks

    raise RuntimeError("Could not generate blocks under all constraints.")

# Generate valid blocks
blocks = create_blocks()

# Flatten into durations and bpms
durations_shuffled, bpms_shuffled = zip(*sum(blocks, []))
parameters = {}
# Assign to parameters
parameters["times"] = np.array(durations_shuffled)
parameters["bpms"] = np.array(bpms_shuffled)
parameters["conditions"] = ["Count"] * len(parameters["times"])

# Optional: print blocks
for i, block in enumerate(blocks, 1):
    print(f"Block {i}: {block}")

print(parameters)  # Print the param