import numpy as np
from scipy import stats
import statistics

# Load data from the text files
#data1 = np.loadtxt("/Users/sofialeksell/FLAdversarialAttacks-git/FLAdversarialAttacks/test_results_centralized.txt")
data1 = np.loadtxt("/Users/sofialeksell/FLAdversarialAttacks-git/FLAdversarialAttacks/experimentCali/test_results_clientadversarial.txt")
data2 = np.loadtxt("/Users/sofialeksell/FLAdversarialAttacks-git/FLAdversarialAttacks/experimentCali/test_results_client.txt")


# Create Boolean masks for negative and positive values
negative_mask_data1 = data1 < 0
negative_mask_data2 = data2 < 0
positive_mask_data1 = data1 > 0
positive_mask_data2 = data2 > 0

# Use the masks to extract negative and positive values
negative_data1 = data1[negative_mask_data1]
negative_data2 = data2[negative_mask_data2]
positive_data1 = data1[positive_mask_data1]
positive_data2 = data2[positive_mask_data2]

# Combine the negative and positive values into a single array
combined_data1 = np.concatenate((positive_data1, np.abs(negative_data1)))
combined_data2 = np.concatenate((positive_data2, np.abs(negative_data2)))

# Calculate the mean of the combined values
mean_combined_data1 = np.mean(combined_data1)
mean_combined_data2 = np.mean(combined_data2)

print(combined_data1)
print("Mean of combined values in data1:", mean_combined_data1)
print(combined_data2)
print("Mean of combined values in data2:", mean_combined_data2)

# Perform a two-sample t-test
t_stat, p_value = stats.ttest_ind(combined_data1, combined_data2)


print(t_stat)
print(p_value)

# Decide on the significance level (alpha)
alpha = 0.05

# Check if the p-value is less than alpha
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between the means.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between the means.")