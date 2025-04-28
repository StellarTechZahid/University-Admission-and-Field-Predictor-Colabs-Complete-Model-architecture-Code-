{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ck0yEOiOwYgZ"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import sklearn as sklearn\n",
        "from sklearn.model_selection import train_test_split, cross_val_score\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, f1_score\n",
        "import joblib\n",
        "import logging\n",
        "import multiprocessing as mp\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Setup logging\n",
        "logging.basicConfig(filename='prediction.log', level=logging.INFO,\n",
        "                    format='%(asctime)s - %(message)s')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Clean column names (remove spaces, newlines)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "df.columns = [col.strip().replace('\\n', '') for col in df.columns]\n",
        "print(\"Columns:\", df.columns.tolist())\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Dynamic university detection (Enhancement 2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "universities = [col.split(' Score')[0] for col in df.columns if ' Score' in col and col not in ['Matriculation Marks', 'Intermediate Marks']]\n",
        "print(\"Detected universities:\", universities)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Define max_marks (aligned with cleaned names)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "max_marks = {\n",
        "    'Matriculation Marks': 1100, 'Intermediate Marks': 550,\n",
        "    'UET Score': 400, 'NUST Score': 200, 'GIKI Score': 100,\n",
        "    'PIEAS Score': 100, 'PUCIT Score': 100, 'FAST Score': 100,\n",
        "    'COMSATS Score': 100, 'ITU Score': 50\n",
        "}\n",
        "feature_cols = ['Matriculation Marks_pct', 'Intermediate Marks_pct']"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Normalize features"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "for col in max_marks.keys():\n",
        "    if col in df.columns:\n",
        "        df[f'{col}_pct'] = np.clip(df[col] / max_marks[col] * 100, 0, 100)\n",
        "        logging.info(f\"Normalized {col} to {col}_pct\")\n",
        "    else:\n",
        "        logging.warning(f\"Column {col} not found in dataset\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Verify all expected columns exist"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "expected_cols = feature_cols + [f'{uni} Score_pct' for uni in universities] + [f'{uni} Aggregate' for uni in universities]\n",
        "missing = [col for col in expected_cols if col not in df.columns]\n",
        "if missing:\n",
        "    logging.error(f\"Missing columns after normalization: {missing}\")\n",
        "    raise ValueError(f\"Normalization failed: {missing}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VvvzFBXShtTx"
      },
      "source": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Verify all expected columns exist"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bCNYiJpnULFt"
      },
      "outputs": [],
      "source": [
        "\n",
        "expected_cols = feature_cols + [f'{uni} Score_pct' for uni in universities] + [f'{uni} Aggregate' for uni in universities]\n",
        "missing = [col for col in expected_cols if col not in df.columns]\n",
        "if missing:\n",
        "    logging.error(f\"Missing columns after normalization: {missing}\")\n",
        "    raise ValueError(f\"Normalization failed: {missing}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Data quality checks (Enhancement 9)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "for uni in universities:\n",
        "    invalid_scores = df[df[f'{uni} Score'] > max_marks[f'{uni} Score']].index\n",
        "    if not invalid_scores.empty:\n",
        "        df.loc[invalid_scores, f'{uni} Score'] = max_marks[f'{uni} Score']\n",
        "        df.loc[invalid_scores, f'{uni} Score_pct'] = 100\n",
        "        logging.info(f\"Capped {len(invalid_scores)} {uni} Scores at {max_marks[f'{uni} Score']}\")\n",
        "    low_agg_high_field = df[(df[f'{uni} Aggregate'] < 60) & (df[f'Program Selected at {uni}'] != 'Low Competitive Fields')].index\n",
        "    if not low_agg_high_field.empty:\n",
        "        logging.warning(f\"Inconsistent data for {uni}: {len(low_agg_high_field)} rows\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Replace 'Not Selected in Any Field'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "\n",
        "for uni in universities:\n",
        "    df.loc[df[f'Program Selected at {uni}'] == 'Not Selected in Any Field', f'Program Selected at {uni}'] = 'Low Competitive Fields'"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Unique Program in each University"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {},
      "outputs": [
        {
          "ename": "NameError",
          "evalue": "name 'universities' is not defined",
          "output_type": "error",
          "traceback": [
            "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "Cell \u001b[1;32mIn[3], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m uni_data \u001b[38;5;241m=\u001b[39m {}\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m uni \u001b[38;5;129;01min\u001b[39;00m \u001b[43muniversities\u001b[49m:\n\u001b[0;32m      3\u001b[0m     features \u001b[38;5;241m=\u001b[39m feature_cols \u001b[38;5;241m+\u001b[39m [\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m{\u001b[39;00muni\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m Score_pct\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m{\u001b[39;00muni\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m Aggregate\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[0;32m      4\u001b[0m     targets \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSelected at \u001b[39m\u001b[38;5;132;01m{\u001b[39;00muni\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m?\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mProgram Selected at \u001b[39m\u001b[38;5;132;01m{\u001b[39;00muni\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m'\u001b[39m]\n",
            "\u001b[1;31mNameError\u001b[0m: name 'universities' is not defined"
          ]
        }
      ],
      "source": [
        "uni_data = {}\n",
        "for uni in universities:\n",
        "    features = feature_cols + [f'{uni} Score_pct', f'{uni} Aggregate']\n",
        "    targets = [f'Selected at {uni}?', f'Program Selected at {uni}']\n",
        "    missing_cols = [col for col in features + targets if col not in df.columns]\n",
        "    if missing_cols:\n",
        "        logging.error(f\"Error: {missing_cols} not found for {uni}\")\n",
        "        continue\n",
        "    uni_df = df[features + targets].copy()\n",
        "    uni_data[uni] = {\n",
        "        'X': uni_df[features],\n",
        "        'y_admission': uni_df[f'Selected at {uni}?'],\n",
        "        'y_program': uni_df[f'Program Selected at {uni}']\n",
        "    }\n",
        "    print(f\"{uni} unique programs:\", uni_data[uni]['y_program'].nunique())\n",
        "if not uni_data:\n",
        "    logging.error(\"No university data prepared—check dataset columns\")\n",
        "    raise ValueError(\"uni_data is empty\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Generating Synthetic Data for Zero Admission Cases"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def generate_synthetic_zeros(df, uni, n_samples=200):\n",
        "    synth_df = pd.DataFrame({\n",
        "        'Matriculation Marks_pct': np.concatenate([np.random.uniform(10, 30, n_samples//2), np.random.uniform(30, 50, n_samples//2)]),\n",
        "        'Intermediate Marks_pct': np.concatenate([np.random.uniform(10, 30, n_samples//2), np.random.uniform(30, 50, n_samples//2)]),\n",
        "        f'{uni} Score_pct': np.concatenate([np.random.uniform(10, 30, n_samples//2), np.random.uniform(30, 50, n_samples//2)]),\n",
        "        f'{uni} Aggregate': np.concatenate([np.random.uniform(20, 40, n_samples//2), np.random.uniform(40, 60, n_samples//2)])\n",
        "    })\n",
        "    synth_df[f'Selected at {uni}?'] = 0\n",
        "    return synth_df"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Training a Machine Learning Model for Admission Predictions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def train_admission_model(uni):\n",
        "    if uni not in uni_data:\n",
        "        logging.error(f\"No data for {uni}\")\n",
        "        return None\n",
        "    real_data = uni_data[uni]['X'].copy()\n",
        "    real_data[f'Selected at {uni}?'] = uni_data[uni]['y_admission']\n",
        "    if real_data.isnull().values.any():\n",
        "        real_data.fillna(real_data.median(), inplace=True)\n",
        "        logging.info(f\"Imputed NaNs for {uni}\")\n",
        "    synth_data = generate_synthetic_zeros(real_data, uni)\n",
        "    model_data = pd.concat([real_data, synth_data], ignore_index=True)\n",
        "\n",
        "    X = model_data.drop(columns=[f'Selected at {uni}?'])\n",
        "    y = model_data[f'Selected at {uni}?']\n",
        "\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\n",
        "    model = LogisticRegression(max_iter=1000)\n",
        "    model.fit(X_train, y_train)\n",
        "\n",
        "    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')\n",
        "    logging.info(f\"{uni} Admission CV F1: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})\")\n",
        "\n",
        "    y_pred = model.predict(X_test)\n",
        "    print(f\"{uni} Admission - Accuracy: {accuracy_score(y_test, y_pred):.2f}, F1: {f1_score(y_test, y_pred):.2f}\")\n",
        "    return model\n",
        "\n",
        "with mp.Pool(processes=mp.cpu_count()) as pool:\n",
        "    admission_models_list = pool.map(train_admission_model, universities)\n",
        "admission_models = {uni: model for uni, model in zip(universities, admission_models_list) if model is not None}\n",
        "\n",
        "for uni, model in admission_models.items():\n",
        "    joblib.dump(model, f'admission_model_{uni}.pkl')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Training a Machine Learning Model for Field  Predictions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def train_program_model(uni):\n",
        "    if uni not in uni_data:\n",
        "        logging.error(f\"No data for {uni}\")\n",
        "        return None\n",
        "    X = uni_data[uni]['X']\n",
        "    y = uni_data[uni]['y_program']\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\n",
        "    model = RandomForestClassifier(n_estimators=200, random_state=42)\n",
        "    model.fit(X_train, y_train)\n",
        "\n",
        "    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')\n",
        "    logging.info(f\"{uni} Program CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})\")\n",
        "\n",
        "    y_pred = model.predict(X_test)\n",
        "    print(f\"{uni} Program - Accuracy: {accuracy_score(y_test, y_pred):.2f}\")\n",
        "    return model\n",
        "\n",
        "with mp.Pool(processes=mp.cpu_count()) as pool:\n",
        "    program_models_list = pool.map(train_program_model, universities)\n",
        "program_models = {uni: model for uni, model in zip(universities, program_models_list) if model is not None}\n",
        "\n",
        "for uni, model in program_models.items():\n",
        "    joblib.dump(model, f'program_model_{uni}.pkl')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "##  Loading Models and Aggregate Formulas"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "admission_models_loaded = {uni: joblib.load(f'admission_model_{uni}.pkl') for uni in universities}\n",
        "program_models_loaded = {uni: joblib.load(f'program_model_{uni}.pkl') for uni in universities}\n",
        "\n",
        "aggregate_formulas = {\n",
        "    'UET': lambda matric_pct, inter_pct, test_pct: (matric_pct * 0.17) + (inter_pct * 0.50) + (test_pct * 0.33),  # Updated UET formula\n",
        "    'FAST': lambda matric_pct, inter_pct, test_pct: (matric_pct * 0.1) + (inter_pct * 0.4) + (test_pct * 0.5),\n",
        "    'NUST': lambda matric_pct, inter_pct, test_pct: (matric_pct * 0.1) + (inter_pct * 0.15) + (test_pct * 0.75),\n",
        "    'GIKI': lambda matric_pct, inter_pct, test_pct: (test_pct * 0.85) + (inter_pct * 0.10) + (5),\n",
        "    'PIEAS': lambda matric_pct, inter_pct, test_pct: (matric_pct * 0.15) + (inter_pct * 0.25) + (test_pct * 0.6),\n",
        "    'PUCIT': lambda matric_pct, inter_pct, test_pct: ((0.25 * (matric_pct * 11 + inter_pct * 5.5)) / (0.25 * (1100 + 570)) * 75) + (test_pct * 0.25),\n",
        "    'COMSATS': lambda matric_pct, inter_pct, test_pct: (matric_pct * 0.1) + (inter_pct * 0.4) + (test_pct * 0.5),\n",
        "    'ITU': lambda matric_pct, inter_pct, test_pct: (test_pct * 0.5) + (inter_pct * 0.35) + (matric_pct * 0.15)\n",
        "}\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Defining the Main Function and Validating Inputs"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def predict_admission_programs(input_dict):\n",
        "    for key, val in input_dict.items():\n",
        "        if 'Marks' in key or 'Score' in key:\n",
        "            max_val = max_marks.get(key, 100)\n",
        "            if not (0 <= val <= max_val):\n",
        "                logging.error(f\"Invalid input: {key} = {val} exceeds {max_val}\")\n",
        "                return {uni: f\"Error: {key} must be between 0 and {max_val}\" for uni in universities}\n",
        "\n",
        "    input_normalized = {\n",
        "        'Matriculation Marks_pct': input_dict['Matriculation Marks'] / 1100 * 100,\n",
        "        'Intermediate Marks_pct': input_dict['Intermediate Marks'] / 550 * 100\n",
        "    }\n",
        "    aggregates = {}\n",
        "    results = {}  # Only include tested universities\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Processing Each University and Predicting Admissions"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "    for uni in universities:\n",
        "        test_flag = input_dict.get(f'Have You Given {uni} Test', 0)\n",
        "        if test_flag == 1:  # Only process tested universities\n",
        "            if f'{uni} Score' not in input_dict:\n",
        "                results[uni] = f\"Error: {uni} Score missing despite test taken\"\n",
        "                aggregates[uni] = None\n",
        "            else:\n",
        "                test_score = input_dict[f'{uni} Score']\n",
        "                input_normalized[f'{uni} Score_pct'] = test_score / max_marks[f'{uni} Score'] * 100\n",
        "                formula = aggregate_formulas.get(uni, aggregate_formulas['UET'])\n",
        "                aggregates[uni] = formula(input_normalized['Matriculation Marks_pct'],\n",
        "                                        input_normalized['Intermediate Marks_pct'],\n",
        "                                        input_normalized[f'{uni} Score_pct'])\n",
        "\n",
        "                uni_input = pd.DataFrame([[\n",
        "                    input_normalized['Matriculation Marks_pct'],\n",
        "                    input_normalized['Intermediate Marks_pct'],\n",
        "                    input_normalized[f'{uni} Score_pct'],\n",
        "                    aggregates[uni]\n",
        "                ]], columns=feature_cols + [f'{uni} Score_pct', f'{uni} Aggregate'])\n",
        "\n",
        "                admission_pred = admission_models_loaded[uni].predict(uni_input)[0]\n",
        "                if admission_pred == 0:\n",
        "                    results[uni] = \"Selected: No, Programs: None\"\n",
        "                else:\n",
        "                    program_pred = program_models_loaded[uni].predict(uni_input)[0]\n",
        "                    results[uni] = f\"Selected: Yes, Program: {program_pred}\"\n",
        "                results[uni] += f\"\\n     Aggregate: {aggregates[uni]:.2f}\"\n",
        "\n",
        "    logging.info(f\"Input: {input_dict}, Results: {results}\")\n",
        "    return results\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Collecting and Validating Basic Student Details"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def main():\n",
        "    print(\"Enter student details:\")\n",
        "    input_dict = {}\n",
        "    try:\n",
        "        # Validate Matriculation and Intermediate Marks\n",
        "        while True:\n",
        "            try:\n",
        "                matric = float(input(\"Matriculation Marks (out of 1100): \"))\n",
        "                if 0 <= matric <= 1100:\n",
        "                    input_dict['Matriculation Marks'] = matric\n",
        "                    break\n",
        "                else:\n",
        "                    print(\"Error: Matriculation Marks must be between 0 and 1100\")\n",
        "            except ValueError:\n",
        "                print(\"Error: Please enter a valid number\")\n",
        "\n",
        "        while True:\n",
        "            try:\n",
        "                inter = float(input(\"Intermediate Marks (out of 550): \"))\n",
        "                if 0 <= inter <= 550:\n",
        "                    input_dict['Intermediate Marks'] = inter\n",
        "                    break\n",
        "                else:\n",
        "                    print(\"Error: Intermediate Marks must be between 550\")\n",
        "            except ValueError:\n",
        "                print(\"Error: Please enter a valid number\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Collecting and Validating University Test Information"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Validate test flags with inline guidance\n",
        "        for uni in universities:\n",
        "            while True:\n",
        "                try:\n",
        "                    prompt = f\"Have you given {uni} test? (Enter 1 for Yes, 0 for No): \"\n",
        "                    test_taken = int(input(prompt))\n",
        "                    if test_taken in [0, 1]:\n",
        "                        input_dict[f'Have You Given {uni} Test'] = test_taken\n",
        "                        break\n",
        "                    else:\n",
        "                        print(f\"Please enter 1 (Yes) or 0 (No) only - invalid input: {test_taken}\")\n",
        "                except ValueError as e:\n",
        "                    print(f\"Please enter 1 (Yes) or 0 (No) only - invalid input: {e}\")\n",
        "\n",
        "            if input_dict[f'Have You Given {uni} Test'] == 1:\n",
        "                while True:\n",
        "                    try:\n",
        "                        score = float(input(f\"{uni} Score (out of {max_marks[f'{uni} Score']}): \"))\n",
        "                        if 0 <= score <= max_marks[f'{uni} Score']:\n",
        "                            input_dict[f'{uni} Score'] = score\n",
        "                            break\n",
        "                        else:\n",
        "                            print(f\"Error: {uni} Score must be between 0 and {max_marks[f'{uni} Score']}\")\n",
        "                    except ValueError:\n",
        "                        print(\"Error: Please enter a valid number\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Predicting and Displaying Admission Results"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "        # Predict and display results\n",
        "        results = predict_admission_programs(input_dict)\n",
        "        print(\"\\nResults:\")\n",
        "        for uni, result in results.items():\n",
        "            print(f\"{uni}: {result}\")\n",
        "    except Exception as e:\n",
        "        print(f\"Error: {str(e)}\")\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# User Testing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "Enter student details:\n",
        "Matriculation Marks (out of 1100): 990\n",
        "Intermediate Marks (out of 550): 449\n",
        "Have you given UET test? (Enter 1 for Yes, 0 for No): 1\n",
        "UET Score (out of 400): 130\n",
        "Have you given NUST test? (Enter 1 for Yes, 0 for No): 1\n",
        "NUST Score (out of 200): 140\n",
        "Have you given GIKI test? (Enter 1 for Yes, 0 for No): 1\n",
        "GIKI Score (out of 100): 67\n",
        "Have you given PIEAS test? (Enter 1 for Yes, 0 for No): 1\n",
        "PIEAS Score (out of 100): 43\n",
        "Have you given PUCIT test? (Enter 1 for Yes, 0 for No): 1\n",
        "PUCIT Score (out of 100): 81\n",
        "Have you given FAST test? (Enter 1 for Yes, 0 for No): 1\n",
        "FAST Score (out of 100): 50\n",
        "Have you given COMSATS test? (Enter 1 for Yes, 0 for No): 1\n",
        "COMSATS Score (out of 100): 67\n",
        "Have you given ITU test? (Enter 1 for Yes, 0 for No): 1\n",
        "ITU Score (out of 50): 32\n",
        "\n",
        "Results:\n",
        "UET: Selected: Yes, Program: City and Regional Planning, Environmental Engineering, Geological Engineering, Transportation Engineering\n",
        "     Aggregate: 66.84\n",
        "NUST: Selected: Yes, Program: Electrical Engineering, Mechanical Engineering, Aerospace Engineering, Bioinformatics\n",
        "     Aggregate: 73.75\n",
        "GIKI: Selected: Yes, Program: Low Competitive Fields\n",
        "     Aggregate: 70.11\n",
        "PIEAS: Selected: Yes, Program: Low Competitive Fields\n",
        "     Aggregate: 59.71\n",
        "PUCIT: Selected: No, Programs: None\n",
        "     Aggregate: 84.88\n",
        "FAST: Selected: Yes, Program: Electrical Engineering, Civil Engineering, BBA, Accounting and Finance, Fintech\n",
        "     Aggregate: 66.65\n",
        "COMSATS: Selected: Yes, Program: Electrical Engineering, BBA, Psychology, Accounting and Finance\n",
        "     Aggregate: 75.15\n",
        "ITU: Selected: Yes, Program: Electrical Engineering, Economics with Data Science, Civil Engineering, Management Sciences\n",
        "     Aggregate: 74.07"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "testenv",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.13"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
