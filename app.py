if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

    if df.shape[1] != len(expected_columns):
        st.error(f"Uploaded CSV format is incorrect! Expected {len(expected_columns)} columns but got {df.shape[1]}.")
    else:
        X_transformed = df  # No preprocessing needed

        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        # Debugging Step: Print SHAP values shape
        st.write(f"SHAP values shape: {np.array(shap_values).shape}")

        if isinstance(shap_values, list):  # For models with multiple classes
            shap_values = shap_values[1]  

        # Ensure SHAP importance matches feature count
        if shap_values.shape[1] == len(expected_columns):
            shap_importance = np.mean(np.abs(shap_values), axis=0)
            importance_df = pd.DataFrame({'Feature': expected_columns, 'SHAP Importance': shap_importance})
            importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)

            # Plot Feature Importance
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            # SHAP Summary Plot
            st.write("### 📊 SHAP Summary Plot")
            shap.summary_plot(shap_values, X_transformed, feature_names=expected_columns, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png")
        else:
            st.error(f"SHAP importance mismatch! Expected {len(expected_columns)} features but got {shap_values.shape[1]}.")
