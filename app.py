if submitted:
    # Create a DataFrame with all 23 features
    user_data = pd.DataFrame([[limit_bal, sex, education, marriage, age,
                               pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                               bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                               pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]],
                             columns=expected_columns)
    
    # Preprocess and predict
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    X_transformed = preprocessor.transform(user_data)
    prediction = classifier.predict(X_transformed)
    probability = classifier.predict_proba(X_transformed)[:, 1]

    st.write("### Prediction Result")
    st.write(f"Default Risk: {'High' if prediction[0] == 1 else 'Low'}")
    st.write(f"Probability of Default: {probability[0]:.2f}")

    # Local SHAP explanation
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    # Check if shap_values is a list (binary classification)
    if isinstance(shap_values, list):
        # For binary classification, use shap_values[1] for the positive class
        shap_values = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        # For non-binary cases, use shap_values directly
        base_value = explainer.expected_value

    # Ensure the input data is in the correct format
    features = user_data.iloc[0:1, :]  # Extract the first row of user data as a DataFrame

    # Generate the SHAP force plot
    st.write("#### Local Explanation (SHAP)")
    shap.force_plot(base_value, shap_values[0], features, matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()
