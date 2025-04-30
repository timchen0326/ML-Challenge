#!/usr/bin/env python3
import re
import json
import numpy as np
import pandas as pd

##############################################
# 1. CLEANING FUNCTIONS 
##############################################

# Mapping dictionaries for text-to-number conversion
word_to_num = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
}

ingredient_descriptors = {'many': 8, 'several': 6, 'few': 3, 'couple': 2}
price_descriptors = {'cheap': 3, 'expensive': 15}

def parse_number_of_ingredients(text, median):
    text = str(text).strip().lower()
    text_clean = re.sub(r'[^a-z0-9\s.\-]', '', text)
    if re.search(r'\d{1,2}[-/]?[a-z]{3,}|[a-z]{3,}[-/]?\d{1,2}', text_clean):
        return median
    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        numbers = [float(num) for num in numbers]
        return sum(numbers) / len(numbers)
    if text_clean in word_to_num:
        return word_to_num[text_clean]
    if text_clean in ingredient_descriptors:
        return ingredient_descriptors[text_clean]
    if ',' in text:
        return len([item for item in text.split(',') if item.strip() != ''])
    return median

def parse_expected_price(text, median):
    text = str(text).strip().lower()
    text_clean = re.sub(r'[^\d.\-]', ' ', text)
    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        numbers = [float(num) for num in numbers]
        return sum(numbers) / len(numbers)
    if text_clean in word_to_num:
        return word_to_num[text_clean]
    if text_clean in price_descriptors:
        return price_descriptors[text_clean]
    return median

# Drink preprocessing
drink_mapping = {
    "carbonated": ["coke", "cola", "cocacola", "coca-cola", "pepsi", "sprite", "7up",
                    "soda", "carbonated soft drink", "soft drink", "pop", "soft", "soft drinks", "fizzy", "fanta",
                     "mountain dew", "carbonated", "jarritos", "ginger ale", "ginger", "canada dry", "gingerale"],
    "iced tea/juice/lemonade": ["iced tea", "ice tea", "nestea", "brisk",
                                "lemonade", "juice" ],
    "water": ["water", "tap water", "sparkling water", "mineral water"],
    "tea": ["tea", "green tea", "hot tea", "oolong tea", "barley tea", "rice tea", "matcha", "jasmine tea", "bubble tea", "boba", "milk tea", "soy"],
    "alcohol": ["beer", "wine", "red wine", "sake", "soju", "rice wine", "champagne", "cocktail", "whiskey", "alcohol"],
    "yogurt drinks": ["ayran", "laban ayran", "laban", "leban", "lassi", "yakult", "yogurt"],
    "soup": ["soup"],
}

def preprocess_drink_name(drink_name):
    if pd.isna(drink_name):
        return None
    drink_name = drink_name.lower().strip()
    drink_name = re.sub(r'[\xa0\n]', ' ', drink_name)
    drink_name = re.sub(r'[^a-z0-9\s]', '', drink_name)
    drink_name = re.sub(r'\s+', ' ', drink_name)
    return drink_name

def standardize_drink(drink_name):
    for standard, variations in drink_mapping.items():
        if any(keyword in drink_name for keyword in variations):
            return standard
    return "other"

# Movie preprocessing
def preprocess_movie_name(movie_name):
    if pd.isna(movie_name):
        return None
    movie_name = movie_name.lower().strip()
    movie_name = re.sub(r'[^a-z0-9\s]', '', movie_name)
    movie_name = re.sub(r'\s+', ' ', movie_name)
    return movie_name

movie_mapping = {
    "cloudy with a chance of meatballs": ["cloudy with a chance of meatballs"],
    "home alone": ["home alone"],
    "spiderman": ["spiderman", "spider man"],
    "avengers": ["avengers", "avenger", "avangers"],
    "ratatouille": ["ratatouille"],
    "charlie and the chocolate factory": ["charlie and the chocolate factory", "willy wonka"],
    "finding nemo": ["finding nemo"],
    "cars": ["cars"],
    "the incredibles": ["incredibles"],
    "moana": ["moana"],
    "beauty and the beast": ["beauty and the beast"],
    "cinderella": ["cinderella"],
    "snow white": ["snow white"],
    "sushi": ["japanese", "tokyo", "japan", "anime", "ghibli", "miyazaki", "shinkai", "sushi", "asian", "asians", "asia",
              "your name", "spirited away"],
    "pizza": ["italian", "napoli", "rome", "mafia", "pazzeria", "pizza", "american",
              "whiplash", "toy story", "ninja turtles", "la la land", "goodfellas"],
    "shawarma": ["pakistani", "pakistan", "shawarma", "middle east", "middle eastern", "india", "indian", "arabia", "turkish", "egypt",
                 "aladdin", "bollywood", "hindi"],
    "not sushi": ["iron man", "ironman", "interstellar", "batman", "the social network", "pulp fiction", "lion king",
                  "mission impossible", "dead reckoning", "deadpool", "dead pool", "hangover", "godfather"],
    "not pizza": ["john wick", "kung fu", "panda", "mulan", "parasite"],
}


def standardize_movie(movie_name):
    for standard, variations in movie_mapping.items():
        if any(keyword in movie_name for keyword in variations):  # Partial match
            return standard
    return "other"  # Assign all unclassified movies to "other"


def process_and_clean_dataset(df):
    # Rename columns as done during training.
    df = df.rename(columns={
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1_complexity',
        'Q2: How many ingredients would you expect this food item to contain?': 'Q2_ingredients',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3_setting',
        'Q4: How much would you expect to pay for one serving of this food item?': 'Q4_price',
        'Q5: What movie do you think of when thinking of this food item?': 'Q5_movie',
        'Q6: What drink would you pair with this food item?': 'Q6_drink',
        'Q7: When you think about this food item, who does it remind you of?': 'Q7_reminds',
        'Q8: How much hot sauce would you add to this food item?': 'Q8_hot_sauce'
    })

    # Compute medians for Q2 and Q4
    median_q2 = df['Q2_ingredients'].apply(lambda x: parse_number_of_ingredients(x, np.nan)).median()
    parsed_prices = df['Q4_price'].apply(lambda x: parse_expected_price(x, np.nan))
    median_q4 = parsed_prices.dropna().median()

    # Create new numeric columns
    df['Q2_new'] = df['Q2_ingredients'].apply(lambda x: parse_number_of_ingredients(x, median_q2))
    df['Q4_new'] = df['Q4_price'].apply(lambda x: parse_expected_price(x, median_q4))
    df['Q5_new'] = df['Q5_movie'].apply(lambda x: standardize_movie(preprocess_movie_name(x)) if pd.notna(x) else "other")
    df['Q6_new'] = df['Q6_drink'].apply(lambda x: standardize_drink(preprocess_drink_name(x)) if pd.notna(x) else "other")
    return df

##############################################
# 2. PREPROCESSING FUNCTIONS
##############################################

def preprocess_raw_data(data, q=5):
    
        if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data, keep_default_na=False)

    # Apply cleaning functions (rename columns, parse numbers, standardize movies and drinks)
    df = process_and_clean_dataset(df)

    # ----- Process multi-hot columns for Q3_setting and Q7_reminds -----
    df['Q3_setting_clean'] = df['Q3_setting'].fillna('').str.strip().str.replace(', ', ',')
    q3_dummies = df['Q3_setting_clean'].str.get_dummies(sep=',')
    q3_dummies.columns = ['Q3_setting_' + c.strip() for c in q3_dummies.columns]

    df['Q7_reminds_clean'] = df['Q7_reminds'].fillna('').str.strip().str.replace(', ', ',')
    q7_dummies = df['Q7_reminds_clean'].str.get_dummies(sep=',')
    q7_dummies.columns = ['Q7_reminds_' + c.strip() for c in q7_dummies.columns]

    df = pd.concat([df, q3_dummies, q7_dummies], axis=1)
    df.drop(columns=['Q3_setting', 'Q3_setting_clean', 'Q7_reminds', 'Q7_reminds_clean'], inplace=True)

    # ----- One-hot encode needed columns -----
    onehot_cols = ['Q1_complexity', 'Q5_new', 'Q6_new', 'Q8_hot_sauce']
    df_encoded = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols)
    df_encoded = df_encoded.astype({col: int for col in df_encoded.columns if df_encoded[col].dtype == bool})

    # ----- Bin real-valued columns and one-hot encode the bins -----
    for col in ['Q2_new', 'Q4_new']:
        df_encoded[col] = pd.qcut(df_encoded[col], q=q, labels=False, duplicates='drop')
        dummies = pd.get_dummies(df_encoded[col], prefix=col)
        df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

    # ----- Final feature list: all columns except 'Label' -----
    feature_cols = [col for col in df_encoded.columns if col not in ["Label", "id"]]
    X = df_encoded[feature_cols]

    return X, feature_cols


##############################################
# 3. STANDARDIZATION FUNCTION
##############################################
def standardize(df, real_cols, means, stds):
    df_scaled = df.copy()
    df_scaled[real_cols] = (df[real_cols] - means) / stds
    return df_scaled

##############################################
# 4. MANUAL NAIVE BAYES FUNCTIONS
##############################################
def fit_naive_bayes(X_train, y_train, alpha=1.0):
    classes = np.unique(y_train)
    n_samples = len(y_train)
    priors = {c: np.sum(y_train == c) / n_samples for c in classes}
    features = X_train.columns
    X_np = X_train.values
    cond_probs = pd.DataFrame(index=classes, columns=features, dtype=float)
    for c in classes:
        X_c = X_np[y_train == c]
        class_sum = X_c.sum(axis=0)
        cond_probs.loc[c] = (class_sum + alpha) / (X_c.shape[0] + alpha * 2)
        cond_probs.loc[c] = cond_probs.loc[c].clip(lower=1e-9, upper=1 - 1e-9)
    return priors, cond_probs

def predict_naive_bayes(X_test, priors, cond_probs):
    classes = cond_probs.index
    X_np = X_test.values
    predictions = []
    log_priors = {c: np.log(p) for c, p in priors.items()}
    log_cond_probs = np.log(cond_probs)
    log_one_minus_cond_probs = np.log(1.0 - cond_probs)
    for i in range(X_test.shape[0]):
        row = X_np[i]
        best_class = None
        best_score = -np.inf
        for c in classes:
            score = log_priors[c]
            cp = log_cond_probs.loc[c].values
            cp_inv = log_one_minus_cond_probs.loc[c].values
            score += np.sum(row * cp + (1 - row) * cp_inv)
            if score > best_score:
                best_score = score
                best_class = c
        predictions.append(best_class)
    return np.array(predictions)

##############################################
# 5. PREDICTION FUNCTION
##############################################
def predict_all(file_path):
    """
    Reads raw test CSV file, applies cleaning and preprocessing exactly as done during training,
    loads the saved model parameters and scaling configuration, and returns predictions.
    """
    # 1. Preprocess raw data (with q=5 for binning real-valued columns if needed)
    q = 5
    X_raw, feature_cols = preprocess_raw_data(file_path, q=q)
    
    # 2. Load scaling parameters and expected feature configuration.
    with open("feature_config.json", "r") as f:
        config = json.load(f)

    real_cols = config.get("real_cols", [])
    expected_features = config["features"]
    means = pd.Series(config.get("means", {}))
    stds = pd.Series(config.get("stds", {}))

    # 3. Standardize only if real_cols exist
    if real_cols:
        X_processed = standardize(X_raw, real_cols, means, stds)
    else:
        X_processed = X_raw.copy()

    # 4. Ensure expected feature alignment
    for col in expected_features:
        if col not in X_processed.columns:
            X_processed[col] = 0
    X_processed = X_processed[expected_features]

    
    # 5. Load model parameters (priors and conditional probabilities)
    # Load priors as Series (robust to all environments)
    model_priors_df = pd.read_csv("model_priors.csv", index_col=0)
    model_priors = model_priors_df.iloc[:, 0].to_dict()
    model_cond_probs = pd.read_csv("model_cond_probs.csv", index_col=0)
    
    # 6. Predict using manual Naive Bayes
    predictions = predict_naive_bayes(X_processed, model_priors, model_cond_probs)
    
    return predictions

##############################################
# 6. MAIN FUNCTION (for testing locally)
##############################################
# def main():
#     # For local testing, change 'test_raw.csv' to your test file path.
#     test_file = "test_raw.csv"
#     preds = predict_all(test_file)
#     # Output predictions (e.g., print to console)
#     print("Predictions:")
#     print(preds)

def main():
    test_file = "test_raw.csv"
    labels_file = "true_labels.txt"

    # Load test data (no label column)
    preds = predict_all(test_file)

    # Load true labels from separate file (one label per line)
    with open(labels_file, "r") as f:
        true_labels = [line.strip() for line in f.readlines()]

    # Print predictions and compare
    print("Predictions:")
    for i, (t, p) in enumerate(zip(true_labels, preds)):
        print(f"Row {i}: True = {t}, Predicted = {p}")

    # Compute accuracy
    accuracy = np.mean(np.array(preds) == np.array(true_labels))
    print(f"\nAccuracy on test set: {accuracy:.4f}")


if __name__ == "__main__":
    main()
