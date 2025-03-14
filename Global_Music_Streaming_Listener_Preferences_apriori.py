from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Load the dataset
file_path = "Global_Music_Streaming_Listener_Preferences.csv"
df = pd.read_csv(file_path)

# # Display basic information about the dataset
# df.info(), df.head()

# Transform data into transactional format
basket = df.groupby(['Streaming Platform', 'Top Genre']).size().unstack(fill_value=0)
# print(basket)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert counts to binary format

print(basket)

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.6, use_colnames=True)

print(frequent_itemsets)

# Generate association rules
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = association_rules(frequent_itemsets, metric="conviction", min_threshold=0.7)

rules = rules[rules['lift'] > 1.2]

print(rules.sort_values(by='lift', ascending=False).head(3))

rules.to_csv("Global_Music_Streaming_Listener_Preferences_Appriori_rules.csv")

# Summarize performance metrics
performance_metrics = {
    "Number of Transactions": len(basket),
    "Number of Unique Product Lines": len(basket.columns),
    "Frequent Itemsets Found": len(frequent_itemsets),
    "Association Rules Generated": len(rules),
}

print(performance_metrics)
