elif app_mode == "📊 Feature Importance":
    st.title("📊 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV file for SHAP Analysis", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV does not match expected format. Please check the number of columns.")
        else:
            with st.spinner("Processing data..."):
                # Extract preprocessor and classifier
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                
                # Transform input data
                X_transformed = preprocessor.transform(df)

                # Get transformed feature names
                feature_names = preprocessor.get_feature_names_out()

                # Use a small sample for faster SHAP computation
                sample_data = X_transformed[:50]

                # SHAP explainability
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(sample_data)

            # Ensure SHAP values are correctly indexed
            correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

            # Fix Shape Mismatch by Ensuring Correct Feature Importance Computation
            try:
                shap_importance = np.abs(correct_shap_values).mean(axis=0).flatten()
                feature_names = np.array(feature_names).flatten()

                if len(shap_importance) != len(feature_names):
                    raise ValueError(f"Mismatch: {len(shap_importance)} SHAP values vs {len(feature_names)} features")

                # Create feature importance dataframe
                importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
                importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

                # Display feature importance table
                st.write("### 🔥 Top 10 Most Important Features")
                st.dataframe(importance_df)

                # Plot feature importance
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
                ax.set_xlabel("SHAP Importance")
                ax.set_ylabel("Feature")
                ax.set_title("📊 Feature Importance")
                plt.gca().invert_yaxis()  # Invert to show highest importance at the top
                st.pyplot(fig)

                # SHAP Summary Plot
                st.write("### 📊 SHAP Summary Plot")
                shap.summary_plot(correct_shap_values, sample_data, feature_names=feature_names, show=False)
                plt.savefig("shap_summary.png", bbox_inches='tight')
                st.image("shap_summary.png")

            except Exception as e:
                st.error(f"Feature importance computation failed: {str(e)}")
