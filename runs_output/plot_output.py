import re
import matplotlib.pyplot as plt

# Path to your log file
filename = "run_50_steps_32_bs"

epochs = []
accuracies = []
numbers = []

with open(filename, "r") as f:
    lines = f.readlines()

for i in range(0, len(lines), 2):
    # First line: "Epoch: 184, Loss: 4.3337, Acc: 0.60"
    match = re.search(r"Epoch:\s*(\d+).*Acc:\s*([\d.]+)", lines[i])
    if match:
        epoch = int(match.group(1))
        acc = float(match.group(2))
        epochs.append(epoch)
        accuracies.append(acc)
    else:
        continue

    # Second line: just the number
    num = float(lines[i + 1].strip())
    numbers.append(num)

# --- Plot 1: Accuracy vs Epoch ---
plt.plot(epochs, accuracies, marker="o", label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("acc.png")


# --- Plot 2: Number vs Epoch ---
plt.plot(epochs, numbers, marker="s", color="orange", label="Number under")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Number vs Epoch")
plt.legend()
plt.grid(True)
plt.savefig("acc_prot.png")