import flwr as fl
import matplotlib.pyplot as plt

#The previous run is saved in the results text file and gets overwritten with new values per round.
#To see the difference between no attack and attack in clients, run no malicious client.py first.

def weighted_average(metrics):
    r2_scores = [num_examples * m['r2_score'] for num_examples, m in metrics]
    mapes = [num_examples * m['mape'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"r2-score": sum(r2_scores) / sum(examples), "mape": sum(mapes) / sum(examples)}

# Define the path to the results text file
results_file = "results.txt"

with open(results_file, "r") as file:
    prev_round_numbers = []
    prev_loss_values = []
    prev_r2_score_values = []
    lines = file.readlines()
    for line in lines:
        if line.startswith("Round"):
            parts = line.split(":")
            round_part = parts[0].strip().split(" ")[1]
            prev_round_numbers.append(int(round_part))
            loss, r2 = map(float, parts[1].strip().split(","))
            prev_loss_values.append(loss)
            prev_r2_score_values.append(r2)

print(prev_loss_values)
print(prev_r2_score_values)

strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
history = fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    )


# Initialize arrays to store test labels and test predictions
all_test_labels = []
all_test_predictions = []

# Loop through client IDs from 1 to 8
for client_id in range(1, 9):
    # Read the test results file for each client
    with open(f"test_results_client{client_id}.txt", "r") as f:
        lines = f.readlines()

    test_labels = []
    test_predictions = []

    for line in lines:
        if line.startswith("Label:"):
            parts = line.split(", ")
            label = float(parts[0].split(": ")[1])
            prediction = float(parts[1].split(": ")[1])
            test_labels.append(label)
            test_predictions.append(prediction)

    # Append the test labels and predictions to the arrays
    all_test_labels.extend(test_labels)
    all_test_predictions.extend(test_predictions)

# Now, you have accumulated test_labels and test_predictions from all clients
print("All Test Labels:", all_test_labels)
print("All Test Predictions:", all_test_predictions)

#Save difference to compare to centralized
#with open(f"test_results_client.txt", "w") as f:
#    for client_id in range(1, 5):
#        for label, prediction in zip(test_labels, test_predictions):
#            difference = label - prediction
#            f.write(f"{difference}\n")

#Save difference to compare to centralized
with open(f"test_results_clientadversarial.txt", "w") as f:
    for client_id in range(1, 9):
        for label, prediction in zip(test_labels, test_predictions):
            difference = label - prediction
            f.write(f"{difference}\n")

# Create a scatter plot for all clients
plt.scatter(all_test_labels, all_test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 70]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Extract loss and R^2 score data
loss_data = history.losses_distributed
r2_score_data = history.metrics_distributed

# Separate the data for plotting
round_numbers = range(1, len(loss_data) + 1)
loss_values = [loss for (_, loss) in loss_data]
r2_score_values = [r2 for _, r2 in r2_score_data['r2-score']]
print(r2_score_values)

# Create the loss plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(round_numbers, loss_values, marker='o', label='Attack', color='red')
plt.plot(round_numbers, prev_loss_values, marker='o', label='No attack', color='blue')
plt.title("Distributed Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()

# Create the R^2 score plot
plt.subplot(2, 1, 2)
plt.plot(round_numbers, r2_score_values, marker='o', label='Attack', color='red')
plt.plot(round_numbers, prev_r2_score_values, marker='o', label='No attack', color='blue')
plt.title("Distributed R^2 Score")
plt.xlabel("Round")
plt.ylabel("R^2 Score")
plt.legend()

plt.tight_layout()
plt.show()

# Save the updated results in the text file
#with open(results_file, "w") as file:
#    for round_num, loss, r2_score in zip(round_numbers, loss_values, r2_score_values):
#        file.write(f"Round {round_num}: {loss}, {r2_score}\n")