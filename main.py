import random

def simulate_betting_data(num_samples, introduce_bias=False):
    """
    Generates simulated betting data (feature, actual_outcome, odds).
    If introduce_bias is True, creates a spurious correlation for overfitting.
    """
    data = []
    for _ in range(num_samples):
        feature = random.choice([0, 1]) # A simple binary feature (e.g., 'home_advantage_indicator')
        
        actual_outcome = random.choice([0, 1]) # Default: truly random outcome (e.g., 'home_win')

        if introduce_bias:
            # Introduce a spurious correlation for the training set:
            # If feature is 1, outcome is more likely to be 1 (e.g., 70% chance).
            # If feature is 0, outcome is more likely to be 0 (e.g., 70% chance).
            # This simulates a pattern that a model might incorrectly learn as significant.
            if feature == 1:
                actual_outcome = 1 if random.random() < 0.7 else 0 # 70% chance of outcome 1
            else: # feature == 0
                actual_outcome = 0 if random.random() < 0.7 else 1 # 70% chance of outcome 0

        # Fixed odds for simplicity (e.g., for a two-way market, both sides 1.90)
        # This implies a bookmaker margin: (1/1.90 + 1/1.90) - 1 = ~5.26%
        odds = {0: 1.90, 1: 1.90} # Odds for predicting outcome 0 or outcome 1
        data.append({'feature': feature, 'actual_outcome': actual_outcome, 'odds': odds})
    return data

def simple_overfit_model(data):
    """
    A 'model' that overfits to a spurious pattern.
    It learns a simple rule: if feature is 1, predict 1; else predict 0.
    This rule is based on the *assumption* that feature=1 strongly correlates with outcome=1,
    which might only be true in a biased training set.
    """
    predictions = []
    for d in data:
        # This rule is 'learned' from observing the spurious correlation in the training data.
        # It's a very simple decision boundary based on a single feature.
        predicted_outcome = 1 if d['feature'] == 1 else 0 # Model predicts outcome based on feature
        predictions.append(predicted_outcome)
    return predictions

def evaluate_bets(data, predictions, bet_amount=1.0):
    """
    Evaluates the profitability and accuracy of bets based on predictions.
    """
    total_profit = 0.0
    correct_predictions = 0
    total_bets = len(data)

    for i in range(total_bets):
        actual = data[i]['actual_outcome']
        predicted = predictions[i]
        
        # We only bet if the model makes a prediction. Here, it always does.
        # We bet on the outcome the model predicts.
        odds_for_our_bet = data[i]['odds'][predicted] 

        if predicted == actual:
            # If prediction is correct, we win (odds - 1) * bet_amount
            total_profit += (odds_for_our_bet - 1) * bet_amount
            correct_predictions += 1
        else:
            # If prediction is incorrect, we lose bet_amount
            total_profit -= bet_amount
    
    accuracy = (correct_predictions / total_bets) * 100 if total_bets > 0 else 0
    return total_profit, accuracy

# --- Main Simulation ---
if __name__ == "__main__":
    print("--- Simulating Betting Model Failure ---")
    print("This example demonstrates how a betting model can appear successful on training data")
    print("due to overfitting to spurious correlations, but fail in real-world scenarios due to")
    print("poor generalization and the inherent bookmaker margin.")

    # 1. Generate Training Data (small, with a spurious bias)
    # This bias simulates a pattern that a model might incorrectly learn as significant.
    N_TRAIN = 50 # Small training set
    train_data = simulate_betting_data(N_TRAIN, introduce_bias=True)
    print(f"\nGenerated {N_TRAIN} training samples with an artificial, spurious bias.")

    # 2. Define a simple, 'overfit' model
    # The model learns a simple rule: if feature=1, predict 1; else predict 0.
    # This rule is 'overfit' because it relies on a pattern that might only exist
    # in the biased training set and doesn't generalize to truly random data.
    train_predictions = simple_overfit_model(train_data)
    train_profit, train_accuracy = evaluate_bets(train_data, train_predictions)

    print(f"\n--- Model Performance on Training Data ({N_TRAIN} samples) ---")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Training Profit/Loss: {train_profit:.2f} units")
    # Expected: High accuracy (around 70%) and positive profit due to the introduced bias.
    # This shows the model "looks good" on data it has seen or data with a similar bias.

    # 3. Generate Test Data (large, truly random, no bias)
    # This represents real-world, unseen data where the spurious pattern does not hold.
    N_TEST = 3671 # Matches the article's context of "3.671 predictions"
    test_data = simulate_betting_data(N_TEST, introduce_bias=False)
    print(f"\nGenerated {N_TEST} test samples (truly random, no spurious bias).")

    # 4. Evaluate the overfit model on Test Data
    test_predictions = simple_overfit_model(test_data)
    test_profit, test_accuracy = evaluate_bets(test_data, test_predictions)

    print(f"\n--- Model Performance on Test Data ({N_TEST} samples) ---")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Profit/Loss: {test_profit:.2f} units")
    # Expected: Accuracy around 50% (random guess) and significant negative profit due to bookmaker margin.
    # This demonstrates the model's failure in a real-world scenario.

    print("\n--- Analysis of Failure ---")
    print("The model achieved high accuracy and positive profit on the training data,")
    print("giving a false sense of success. However, it failed significantly on unseen test data.")
    print("This failure is due to two main reasons:")
    print("1. Overfitting: The model learned a spurious pattern from the biased training data")
    print("   (e.g., 'if feature=1, outcome is likely 1') that did not generalize to truly random data.")
    print("2. Bookmaker Margin: Even with an accuracy close to 50% (like random guessing),")
    print("   the fixed odds (1.90 for both outcomes) mean that each correct bet wins 0.90 units,")
    print("   while each incorrect bet loses 1.00 unit. This inherent disadvantage (the bookmaker's margin,")
    print("   approximately 5.26% in this case) quickly leads to losses without a strong, generalized predictive edge.")
    print("This highlights why models that 'look good' on historical data often fail to be profitable in real betting.")
