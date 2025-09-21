import numpy as np

# --- Setup ---
# Inputs (x1, x2)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# Target outputs for x1 NOR x2
targets = np.array([1, 0, 0, 0])

# Add bias input (x0 = 1)
X = np.insert(inputs, 0, 1, axis=1)

# --- Training Parameters ---
weights = np.array([0.0, 0.0, 0.0])
learning_rate = 1
max_epochs = 10 # Set a max number of epochs to prevent infinite loops

# --- Table Header ---
header = "| Epoch | x₁ | x₂ | t | x = (1, x₁, x₂) | Current w = (w₀, w₁, w₂) | w · x | y | e | Δw = e · x      | New w = (w₀, w₁, w₂) |"
separator = "|:-----:|:--:|:--:|:-:|:---------------:|:-------------------------:|:-----:|:-:|:-:|:----------------:|:---------------------:|"
print(header)
print(separator)

# Initial state print
initial_w_str = f"({weights[0]:.0f}, {weights[1]:.0f}, {weights[2]:.0f})"
print(f"|       |    |    |   |                 | **{initial_w_str}** |       |   |   |                  |                       |")


# --- Training Loop ---
for epoch in range(1, max_epochs + 1):
    updates_in_epoch = 0
    for i in range(len(X)):
        x_vec = X[i]
        target = targets[i]
        
        # Store current weights for printing
        current_w_str = f"({weights[0]:.0f}, {weights[1]:.0f}, {weights[2]:.0f})"
        
        # 1. Calculate net input
        net_input = np.dot(weights, x_vec)
        
        # 2. Apply step function
        y = 1 if net_input >= 0 else 0
        
        # 3. Calculate error
        error = target - y
        
        # 4. Calculate weight update
        delta_w = learning_rate * error * x_vec
        
        # 5. Update weights
        if error != 0:
            updates_in_epoch += 1
        weights += delta_w
        
        # --- Print table row ---
        x_str = f"({x_vec[0]}, {x_vec[1]}, {x_vec[2]})"
        delta_w_str = f"({delta_w[0]:.0f}, {delta_w[1]:.0f}, {delta_w[2]:.0f})"
        new_w_str = f"({weights[0]:.0f}, {weights[1]:.0f}, {weights[2]:.0f})"

        print(f"| **{epoch}** | {x_vec[1]}  | {x_vec[2]}  | {target} | {x_str: <15} | {current_w_str: <25} | {net_input: >5.0f} | {y} | {error: >1.0f} | {delta_w_str: <16} | {new_w_str: <21} |")

    # Check for convergence
    if updates_in_epoch == 0:
        print(f"\nConvergence reached in Epoch {epoch}. No further updates.")
        break