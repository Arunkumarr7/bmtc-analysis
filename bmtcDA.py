import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import trim_mean, pearsonr

# --- Page Configuration ---
st.set_page_config(page_title="BMTC Statistical Analysis", layout="wide")
st.title("🚌 BMTC Financial Statistical Report")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload BMTC Financial CSV", type="csv")

if uploaded_file is not None:
    # --- Data Loading & Preview ---
    st.header("1. Data Preview & Inconsistency Check")
    df_raw = pd.read_csv(uploaded_file)
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        st.subheader("Raw Sample (First 5 Rows)")
        st.dataframe(df_raw.head())
    with col_pre2:
        st.subheader("Missing Values Check")
        st.write(df_raw.isnull().sum())
    
    # --- Data Cleaning Logic ---
    target_factors = ["Through Sale of Tickets", "Monthly pass", "Daily pass", "Student pass", "Others", "Total"]
    df_raw['Factors_clean'] = df_raw['Factors'].str.strip().str.lower()
    
    selected = {}
    for t in target_factors:
        # Match rows based on target factors
        match = df_raw[df_raw['Factors_clean'].str.contains(t.lower().split()[0], na=False)]
        if len(match) > 0: 
            selected[t] = match.iloc[0]
    
    df_sel = pd.DataFrame(selected).T
    # Extract only the year columns (e.g., 2018-19, 2022-23)
    year_cols = [c for c in df_sel.columns if ("20" in c or "19" in c) and "bifurcation" not in c]
    
    # Convert string numbers (with commas) to floats
    df = df_sel[year_cols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", "", regex=False), errors="coerce")).T
    df = df.dropna(how='all').fillna(0)
    df.index.name = "Year"

    st.subheader("Cleaned & Transposed Dataset (Lakhs)")
    st.dataframe(df)

    # --- 2. Summary Statistics ---
    st.header("2. Summary Statistics")
    stats_dict = {
        "Mean": df.mean(),
        "Median": df.median(),
        "Trimmed Mean (10%)": df.apply(lambda x: trim_mean(x, 0.1)),
        "Std Dev": df.std(),
        "MAD": df.apply(lambda x: (x - x.mean()).abs().mean()),
        "IQR": df.quantile(0.75) - df.quantile(0.25)
    }
    st.dataframe(pd.DataFrame(stats_dict).T)

    # --- 3. Factor Analysis (Trend, Box, Hist & Q-Q) ---
    st.header("3. Specific Factor Analysis")
    selected_col = st.selectbox("Select a Factor to Analyze:", df.columns)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.subheader("Line Trend")
        fig1, ax1 = plt.subplots()
        ax1.plot(df.index, df[selected_col], marker='o', color='blue', linewidth=2)
        ax1.set_ylabel("Amount (Lakhs)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.info("Visualizes growth or decline over the years.")
    
    with col_b:
        st.subheader("Box Plot")
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=df[selected_col], ax=ax2, color="lightgreen")
        ax2.set_ylabel("Lakhs")
        st.pyplot(fig2)
        st.info("Identifies outliers and data spread.")
        
    with col_c:
        st.subheader("Histogram")
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(df[selected_col], bins=8, color='skyblue', edgecolor='black')
        ax_hist.set_xlabel("Lakhs")
        ax_hist.set_ylabel("Frequency")
        st.pyplot(fig_hist)
        st.info("Shows the distribution frequency of revenue values.")

    with col_d:
        st.subheader("Q-Q Plot")
        fig3, ax3 = plt.subplots()
        stats.probplot(df[selected_col], dist="norm", plot=ax3)
        st.pyplot(fig3)
        st.info("If dots follow the red line, the data is Normally Distributed.")

    # --- 4. Correlation Matrix ---
    st.header("4. Correlation Matrix")
    corr = df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # --- 5. Hypothesis Testing ---
    st.header("5. Null Hypothesis Testing")
    st.markdown("""
    * **Null Hypothesis ($H_0$):** There is **no** significant linear relationship between these factors.
    * **Alternative Hypothesis ($H_a$):** There **is** a significant linear relationship.
    """)
    
    col_x = st.selectbox("Select Independent Variable (X)", df.columns, index=min(2, len(df.columns)-1))
    col_y = st.selectbox("Select Dependent Variable (Y)", df.columns, index=min(5, len(df.columns)-1))
    
    if col_x != col_y:
        coeff, p_value = pearsonr(df[col_x], df[col_y])
        
        st.subheader("Statistical Results")
        st.write(f"**Correlation Coefficient (r):** {coeff:.4f}")
        st.write(f"**P-Value:** {p_value:.4f}")
        
        if p_value < 0.05:
            st.success(f"**Conclusion:** Reject $H_0$. Statistically significant relationship at 5% level.")
        else:
            st.error(f"**Conclusion:** Fail to Reject $H_0$. Relationship is not statistically significant (p > 0.05).")
    else:
        st.warning("Please choose two different variables.")
        p_value = 1.0 # Default for conclusion logic

    # --- 6. Relationship Visuals ---
    st.header("6. Relationship Visualization")
    tab1, tab2 = st.tabs(["Scatter Plot", "Violin Plot"])
    
    with tab1:
        st.subheader("Scatter Plot with Regression Line")
        fig4, ax4 = plt.subplots()
        sns.regplot(x=df[col_x], y=df[col_y], ax=ax4, color="purple")
        ax4.set_xlabel(f"{col_x} (Lakhs)")
        ax4.set_ylabel(f"{col_y} (Lakhs)")
        st.pyplot(fig4)

    with tab2:
        st.subheader("Violin Plot (Density & Distribution)")
        fig5, ax5 = plt.subplots()
        sns.violinplot(data=df[[col_x, col_y]], ax=ax5)
        ax5.set_ylabel("Amount (Lakhs)")
        st.pyplot(fig5)

    # --- 7. Conclusion ---
    st.header("7. Conclusion")
    sig_text = "significant" if p_value < 0.05 else "not statistically significant"
    st.write(f"""
    * **Data Consistency:** The dataset was successfully cleaned. Commas were removed and years transposed correctly.
    * **Trend Observation:** The **Line Chart** for **{selected_col}** shows the financial trajectory over the recorded period.
    * **Normality:** Based on the Q-Q plot and Histogram, we can observe if **{selected_col}** follows a Gaussian distribution.
    * **Hypothesis:** For the pair **{col_x}** and **{col_y}**, the p-value is **{p_value:.4f}**, indicating the relationship is **{sig_text}**.
    """)

else:
    st.info("Awaiting CSV file upload...")
